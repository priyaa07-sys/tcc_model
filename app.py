
import streamlit as st
import h5py, io, tempfile, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage
from pathlib import Path

st.set_page_config(
    page_title="TCC Detector | INSAT-3D",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

THRESHOLDS = {
    'North Indian Ocean': {'lat_range':(0,30),  'lon_range':(30,100),'tb_thresh':218.0},
    'South Indian Ocean': {'lat_range':(-30,0), 'lon_range':(30,110),'tb_thresh':221.0},
}
TCC = {'min_radius_km':111.,'min_area_km2':34800.,'independence_km':1200.,'pixel_res_km':4.}

@st.cache_data(show_spinner=False)
def load_insat(filepath):
    with h5py.File(filepath,'r') as f:
        raw  = f['IMG_TIR1'][0].astype(np.uint16)
        lut  = f['IMG_TIR1_TEMP'][:]
        fill = int(f['IMG_TIR1'].attrs['_FillValue'][0])
        tb   = lut[raw.astype(np.int32)].astype(np.float32)
        tb[raw>=fill] = np.nan
        X,Y  = f['X'][:], f['Y'][:]
        lon_0= float(f.attrs['Nominal_Central_Point_Coordinates(degrees)_Latitude_Longitude'][1])
        R    = 6_371_000.
        lons = np.degrees(X/R)+lon_0
        lats = np.degrees(2*np.arctan(np.exp(Y/R))-np.pi/2)
        lon2d= np.tile(lons[None,:],(len(Y),1)).astype(np.float32)
        lat2d= np.tile(lats[:,None],(1,len(X))).astype(np.float32)
        meta = {
            'date': f.attrs.get('Acquisition_Date',b'').decode(),
            'time': f.attrs.get('Acquisition_Time_in_GMT',b'').decode(),
            'name': f.attrs.get('HDF_Product_File_Name',b'').decode(),
        }
    return tb, lat2d, lon2d, meta

def haversine(lat1,lon1,lat2,lon2):
    R=6371.; p1,p2=np.radians(lat1),np.radians(lat2)
    dp=np.radians(lat2-lat1); dl=np.radians(lon2-lon1)
    a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2
    return 2*R*np.arcsin(np.clip(np.sqrt(a),0,1))

def detect_tccs(tb,lat,lon,thresholds):
    cold=np.zeros(tb.shape,bool)
    for p in thresholds.values():
        r=((lat>=p['lat_range'][0])&(lat<=p['lat_range'][1])&
           (lon>=p['lon_range'][0])&(lon<=p['lon_range'][1]))
        cold|=r&(tb<p['tb_thresh'])
    labeled,n=ndimage.label(cold,structure=np.ones((3,3),int))
    px=TCC['pixel_res_km']**2; tccs=[]
    for cid in range(1,n+1):
        mask=labeled==cid; area=mask.sum()*px
        if area<TCC['min_area_km2']: continue
        rows,cols=np.where(mask); plat,plon=lat[rows,cols],lon[rows,cols]
        clat,clon=plat.mean(),plon.mean()
        dists=haversine(clat,clon,plat,plon)
        if dists.max()<TCC['min_radius_km']: continue
        tbv=tb[mask]; mi=np.argmin(tbv)
        tccs.append(dict(
            id=cid,
            clat=float(lat[rows[mi],cols[mi]]),
            clon=float(lon[rows[mi],cols[mi]]),
            n_pix=int(mask.sum()), area=float(area),
            min_tb=float(tbv.min()), mean_tb=float(tbv.mean()),
            median_tb=float(np.median(tbv)), std_tb=float(tbv.std()),
            max_r=float(dists.max()), min_r=float(dists.min()),
            mean_r=float(dists.mean()),
            max_cth=float((300-tbv.min())/6.5),
            mean_cth=float((300-tbv.mean())/6.5),
            mask=mask,
        ))
    return tccs

#  U-Net Model 
class DoubleConv(nn.Module):
    def __init__(self,ic,oc,mc=None):
        super().__init__(); mc=mc or oc
        self.b=nn.Sequential(
            nn.Conv2d(ic,mc,3,padding=1,bias=False),nn.BatchNorm2d(mc),nn.ReLU(True),
            nn.Conv2d(mc,oc,3,padding=1,bias=False),nn.BatchNorm2d(oc),nn.ReLU(True))
    def forward(self,x): return self.b(x)

class Down(nn.Module):
    def __init__(self,ic,oc):
        super().__init__(); self.n=nn.Sequential(nn.MaxPool2d(2),DoubleConv(ic,oc))
    def forward(self,x): return self.n(x)

class Up(nn.Module):
    def __init__(self,ic,oc,bilinear=True):
        super().__init__()
        self.up=(nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
                 if bilinear else nn.ConvTranspose2d(ic,ic//2,2,stride=2))
        self.conv=DoubleConv(ic,oc,ic//2)
    def forward(self,x,skip):
        x=self.up(x)
        dH=skip.shape[2]-x.shape[2]; dW=skip.shape[3]-x.shape[3]
        x=F.pad(x,[dW//2,dW-dW//2,dH//2,dH-dH//2])
        return self.conv(torch.cat([skip,x],1))

class TCCUNet(nn.Module):
    def __init__(self,f=32):
        super().__init__()
        self.inc=DoubleConv(1,f); self.d1=Down(f,f*2); self.d2=Down(f*2,f*4)
        self.d3=Down(f*4,f*8); self.d4=Down(f*8,f*8)
        self.u1=Up(f*16,f*4); self.u2=Up(f*8,f*2); self.u3=Up(f*4,f); self.u4=Up(f*2,f)
        self.outc=nn.Conv2d(f,1,1)
    def forward(self,x):
        x1=self.inc(x); x2=self.d1(x1); x3=self.d2(x2); x4=self.d3(x3); x5=self.d4(x4)
        x=self.u1(x5,x4); x=self.u2(x,x3); x=self.u3(x,x2); x=self.u4(x,x1)
        return torch.sigmoid(self.outc(x))

@st.cache_resource
def load_model(path):
    m=TCCUNet()
    if path and Path(path).exists():
        m.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
        st.sidebar.success("✅ CNN weights loaded")
    else:
        st.sidebar.warning("⚠️ No trained weights — CNN is random (traditional algo still works)")
    m.eval(); return m

def cnn_predict(model,tbn,thresh,P=256,S=128):
    H,W=tbn.shape; prob=np.zeros((H,W),np.float32); cnt=np.zeros((H,W),np.float32)
    with torch.no_grad():
        for i in range(0,max(H-P+1,1),S):
            for j in range(0,max(W-P+1,1),S):
                i2,j2=min(i+P,H),min(j+P,W); pi,pj=i2-i,j2-j
                tile=np.zeros((P,P),np.float32); tile[:pi,:pj]=tbn[i:i2,j:j2]
                t=torch.from_numpy(tile).unsqueeze(0).unsqueeze(0)
                out=model(t).squeeze().numpy()
                prob[i:i2,j:j2]+=out[:pi,:pj]; cnt[i:i2,j:j2]+=1.
    cnt[cnt==0]=1; return prob/cnt

def fig_tb(tb, tccs, meta):
    fig,ax=plt.subplots(figsize=(10,8))
    ax.imshow(tb, cmap='RdYlBu', vmin=190, vmax=310,
              extent=[0,tb.shape[1],tb.shape[0],0])
    ov=np.zeros((*tb.shape,4),np.float32)
    for t in tccs: ov[t['mask']]=[1,.3,0,.6]
    ax.imshow(ov, extent=[0,tb.shape[1],tb.shape[0],0])
    for t in tccs:
        ax.plot(t['clon_px'],t['clat_px'],'w+',ms=14,mew=2.5)
        ax.annotate(f"TCC#{t['id']}",(t['clon_px'],t['clat_px']),
                    color='white',fontsize=8,
                    textcoords='offset points',xytext=(6,6))
    ax.set_title(f"Brightness Temperature + TCC Detections\n"
                 f"{meta['date']} {meta['time']} UTC", fontsize=11, fontweight='bold')
    ax.axis('off'); plt.tight_layout()
    return fig

def fig_cnn(prob, cnn_mask):
    fig,ax=plt.subplots(figsize=(10,8))
    im=ax.imshow(prob,cmap='plasma',vmin=0,vmax=1)
    ax.contour(cnn_mask,levels=[.5],colors='cyan',linewidths=1.5)
    plt.colorbar(im,ax=ax,label='CNN Probability',fraction=.03)
    ax.set_title('CNN TCC Probability Map\n(cyan contour = detection boundary)',
                 fontsize=11, fontweight='bold')
    ax.axis('off'); plt.tight_layout()
    return fig

# UI 
st.markdown("""
<style>
.big-title{font-size:2.8rem;font-weight:800;color:#1a73e8;margin-bottom:0}
.sub{font-size:1.05rem;color:#444;margin-top:0}
.metric-box{background:#f0f4ff;border-radius:10px;padding:12px 18px;margin:6px 0}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title"> TCC Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Tropical Cloud Cluster Detection from INSAT-3D | </p>', unsafe_allow_html=True)
st.divider()

# Sidebar
#with st.sidebar:
    #st.image("https://www.isro.gov.in/media/isro-image/logo-new-header.png",
            # width=120, caption="ISRO")
st.header("⚙️ Configuration & Upload")

uploaded = st.file_uploader(
        "Upload INSAT-3D .h5 file",
        type=['h5','hdf5','hdf'],
        help="INSAT-3D L1C HDF5 file (e.g. 3DIMG_*_L1C_*.h5)"
)

if not uploaded:
    col1,col2,col3 = st.columns(3)
    col1.info("**Step 1** — Upload an INSAT-3D L1C `.h5` file using the sidebar.")
    col2.info("**Step 2** — Adjust thresholds and CNN settings.")
    col3.info("**Step 3** — Click **Detect TCCs** ")

st.subheader("Brightness Temp Thresholds")
nio_thresh = st.slider("North Indian Ocean (K)", 200, 230, 218)
sio_thresh = st.slider("South Indian Ocean (K)", 200, 235, 221)

st.subheader("CNN Settings")
model_path = st.text_input("Model weights path", "tcc_unet.pth")
cnn_thresh = st.slider("CNN probability threshold", 0.1, 0.9, 0.45, 0.05)
use_cnn    = st.checkbox("Enable CNN prediction", value=True)

st.subheader("Minimum Cluster Criteria")
min_area   = st.number_input("Min area (km²)",   value=34800, step=1000)
min_radius = st.number_input("Min radius (km)",  value=111,   step=10)

run_btn = st.button("🔍 Detect TCCs", type="primary", use_container_width=True)

# Update thresholds
user_thresholds = {
    'North Indian Ocean': {'lat_range':(0,30),  'lon_range':(30,100),'tb_thresh':float(nio_thresh)},
    'South Indian Ocean': {'lat_range':(-30,0), 'lon_range':(30,110),'tb_thresh':float(sio_thresh)},
}
TCC['min_area_km2']   = float(min_area)
TCC['min_radius_km']  = float(min_radius)

# Main panel

    #st.markdown("""
    ### Algorithm Overview
    #| Step | Method | Purpose |
    #|------|--------|---------|
    #| 1 | LUT calibration | DN → Brightness Temperature (K) |
    #| 2 | Basin threshold | Isolate cold-cloud pixels (<218 K / <221 K) |
    #| 3 | Connected blobs | Group cold pixels into clusters |
    #| 4 | Area + Radius filter | Remove non-convective structures |
    #| 5 | U-Net CNN | Probabilistic TCC detection map |
    #| 6 | Independence check | Separate TCCs >1200 km apart |
    #| 7 | Parameter extraction | All required TCC metrics |
    #""")

if uploaded and run_btn:
    with st.spinner("Loading INSAT-3D data …"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
        data = uploaded.getvalue()
        tmp.write(data)
        tmp.flush()
        tmp.close()
        tb, lat, lon, meta = load_insat(tmp.name)
        st.success(f"✅ Loaded: **{meta['name']}**  |  {meta['date']} {meta['time']} UTC")

    valid = tb[~np.isnan(tb)]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Image Shape",   f"{tb.shape[0]} × {tb.shape[1]}")
    c2.metric("Min Tb (K)",    f"{valid.min():.1f}")
    c3.metric("Max Tb (K)",    f"{valid.max():.1f}")
    c4.metric("Valid Pixels",  f"{len(valid):,}")

    with st.spinner("Running traditional TCC algorithm …"):
        tccs = detect_tccs(tb, lat, lon, user_thresholds)

    # Add pixel coordinates for plotting
    for t in tccs:
        t['clat_px'] = int(np.argmin(np.abs(lat[:,0]-t['clat'])))
        t['clon_px'] = int(np.argmin(np.abs(lon[0,:]-t['clon'])))

    # CNN
    prob_map = None
    if use_cnn:
        with st.spinner("Running CNN inference (sliding window) …"):
            model = load_model(model_path)
            tbn = np.clip((tb-170.)/(340.-170.),0,1)
            tbn = np.where(np.isnan(tbn),0.,tbn).astype(np.float32)
            prob_map = cnn_predict(model, tbn, cnn_thresh)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗺️ Detections", "📊 CNN Map", "📋 TCC Table", "📥 Export"]
    )

    with tab1:
        fig = fig_tb(tb, tccs, meta)
        st.pyplot(fig, use_container_width=True)
        st.caption(f"Orange regions = detected TCCs  |  White + = convective centre")

    with tab2:
        if prob_map is not None:
            fig2 = fig_cnn(prob_map, prob_map > cnn_thresh)
            st.pyplot(fig2, use_container_width=True)
            st.metric("CNN-detected area (pixels)", int((prob_map>cnn_thresh).sum()))
        else:
            st.info("Enable CNN prediction to see the CNN map.")

    with tab3:
        st.subheader(f"Detected TCCs: {len(tccs)}")
        if tccs:
            rows = []
            for t in tccs:
                rows.append({
                    'ID': t['id'],
                    'Conv Lat (°)': round(t['clat'],3),
                    'Conv Lon (°)': round(t['clon'],3),
                    'Pixel Count':  t['n_pix'],
                    'Min Tb (K)':   round(t['min_tb'],1),
                    'Mean Tb (K)':  round(t['mean_tb'],1),
                    'Median Tb (K)':round(t['median_tb'],1),
                    'Std Tb (K)':   round(t['std_tb'],2),
                    'Max R (km)':   round(t['max_r'],1),
                    'Min R (km)':   round(t['min_r'],1),
                    'Mean R (km)':  round(t['mean_r'],1),
                    'Area (km²)':   round(t['area'],0),
                    'Max CTH (km)': round(t['max_cth'],2),
                    'Mean CTH (km)':round(t['mean_cth'],2),
                })
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            # Per-TCC 
            for t in tccs:
                with st.expander(f"🌀 TCC #{t['id']} — ({t['clat']:.2f}°, {t['clon']:.2f}°)"):
                    cc=st.columns(4)
                    cc[0].metric("Min Tb",    f"{t['min_tb']:.1f} K")
                    cc[1].metric("Mean Tb",   f"{t['mean_tb']:.1f} K")
                    cc[2].metric("Max Radius",f"{t['max_r']:.0f} km")
                    cc[3].metric("Area",      f"{t['area']:.0f} km²")
                    cc2=st.columns(3)
                    cc2[0].metric("Pixels",   t['n_pix'])
                    cc2[1].metric("Max CTH",  f"{t['max_cth']:.1f} km")
                    cc2[2].metric("Mean CTH", f"{t['mean_cth']:.1f} km")
        else:
            st.warning("No TCCs detected with current settings. "
                       "Try adjusting the thresholds in the sidebar.")

    with tab4:
        st.subheader("Export Results")
        if tccs:
            import json
            export = []
            for t in tccs:
                row = {k:v for k,v in t.items() if k!='mask'}
                export.append(row)
            json_str = json.dumps({'metadata':meta,'tccs':export}, indent=2)
            st.download_button("⬇️ Download JSON", json_str,
                               file_name="tcc_detections.json",
                               mime="application/json")
            if prob_map is not None:
                buf = io.BytesIO()
                np.save(buf, prob_map); buf.seek(0)
                st.download_button("⬇️ Download CNN Prob Map (.npy)", buf,
                                   file_name="cnn_prob_map.npy",
                                   mime="application/octet-stream")
        st.info("For NetCDF output, add `netCDF4` to requirements and extend the export function.")

    if os.path.exists(tmp.name):
     os.remove(tmp.name)

# Footer
st.divider()
st.caption("TCC Detection System | Tropical Cloud Cluster Detection from INSAT-3D ")

#with st.spinner("Loading INSAT-3D data …"):
    #tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.h5')
    
    #data = uploaded.getvalue()
    #tmp.write(data)
    #tmp.flush()
    #tmp.close()          # ← IMPORTANT

    #tb, lat, lon, meta = load_insat(tmp.name)
    
#if os.path.exists(tmp.name):
    #os.remove(tmp.name)
    

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import linregress
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, ExponentialSmoothing

st.set_page_config(page_title='Forecasting', page_icon='🔮', layout='wide')
st.markdown('<style>[data-testid="stSidebar"] {display:none;}</style>', unsafe_allow_html=True)
if st.button('⬅️ กลับหน้าหลัก'):
    st.switch_page('Home.py')

st.title('🔮 การพยากรณ์ความต้องการการขนส่ง')
st.write('ระบบตรวจสอบลักษณะอนุกรมเวลา คัดเลือกวิธีที่เหมาะสม และเปรียบเทียบความคลาดเคลื่อนจากการพยากรณ์ย้อนหลัง')
st.divider()

# ---------- DATA ----------
if 'data' not in st.session_state:
    st.warning('⚠️ ยังไม่มีข้อมูล กรุณากลับหน้าหลักและอัปโหลดไฟล์ก่อน')
    st.stop()
raw = st.session_state['data'].copy()

# case-insensitive column matching
colmap = {str(c).strip().lower(): c for c in raw.columns}
time_candidates = [colmap[x] for x in ('date','month','year') if x in colmap]
demand_col = colmap.get('demand')
if not time_candidates or demand_col is None:
    st.error('❌ ไฟล์ต้องมีคอลัมน์ Demand และคอลัมน์เวลาอย่างน้อย 1 คอลัมน์ ได้แก่ Date, Month หรือ Year')
    st.stop()
if len(time_candidates) > 1:
    time_col = st.selectbox('พบคอลัมน์เวลามากกว่า 1 คอลัมน์ กรุณาเลือกคอลัมน์ที่ต้องการใช้', time_candidates)
else:
    time_col = time_candidates[0]

# Parse Year safely; otherwise normal datetime parsing.
time_name = str(time_col).strip().lower()
if time_name == 'year':
    yrs = pd.to_numeric(raw[time_col], errors='coerce')
    parsed_time = pd.to_datetime(yrs.astype('Int64').astype(str) + '-01-01', errors='coerce')
else:
    parsed_time = pd.to_datetime(raw[time_col], errors='coerce')

demand = pd.to_numeric(raw[demand_col], errors='coerce')
invalid = parsed_time.isna() | demand.isna()
if invalid.any():
    st.error(f'❌ พบข้อมูล Date/Month/Year หรือ Demand ที่อ่านไม่ได้ {int(invalid.sum())} แถว กรุณาแก้ไฟล์ต้นฉบับก่อนพยากรณ์')
    st.stop()

df = pd.DataFrame({'Time': parsed_time, 'Demand': demand.astype(float)})
if df['Time'].duplicated().any():
    dups = df.loc[df['Time'].duplicated(keep=False), 'Time'].dt.strftime('%Y-%m-%d').unique()
    st.error(f"❌ พบช่วงเวลาซ้ำ {len(dups)} ช่วง เช่น {', '.join(dups[:5])} กรุณารวม/แก้ข้อมูลก่อน")
    st.stop()
df = df.sort_values('Time').reset_index(drop=True)
if len(df) < 4:
    st.error('❌ ควรมีข้อมูลอย่างน้อย 4 ช่วงเวลา')
    st.stop()

# ---------- FREQUENCY ----------
def classify_frequency(freq):
    if not freq: return None
    f = str(freq).upper()
    if f == 'D': return 'Daily'
    if f.startswith('W') or f == '7D': return 'Weekly'
    if f in ('MS','M','ME'): return 'Monthly'
    if f.startswith(('Q','QS','QE')): return 'Quarterly'
    if f.startswith(('Y','A','YS','YE')): return 'Yearly'
    return None

try:
    inferred_freq = pd.infer_freq(df['Time'])
except Exception:
    inferred_freq = None
frequency_type = classify_frequency(inferred_freq)
opts = {'รายวัน (Daily)':'Daily','รายสัปดาห์ (Weekly)':'Weekly','รายเดือน (Monthly)':'Monthly','รายไตรมาส (Quarterly)':'Quarterly','รายปี (Yearly)':'Yearly'}
labels = {v:k for k,v in opts.items()}
st.subheader('⏱️ ความถี่ของข้อมูล')
if frequency_type is None:
    st.warning('⚠️ ระบบระบุความถี่อัตโนมัติไม่ได้ กรุณาเลือกความถี่ที่ถูกต้อง')
    frequency_type = opts[st.selectbox('เลือกความถี่ของข้อมูล', list(opts))]
else:
    st.info(f'ระบบตรวจพบ: **{labels[frequency_type]}**')

period_map = {'Daily':7,'Weekly':52,'Monthly':12,'Quarterly':4,'Yearly':None}
seasonal_period = period_map[frequency_type]

# Check gaps against selected/inferred frequency.
def expected_dates(start, end, freq_type, inferred):
    if inferred and classify_frequency(inferred) == freq_type:
        try: return pd.date_range(start, end, freq=inferred)
        except Exception: pass
    if freq_type == 'Daily': return pd.date_range(start, end, freq='D')
    if freq_type == 'Weekly': return pd.date_range(start, end, freq='7D')
    if freq_type == 'Monthly': return pd.date_range(start, end, freq='MS' if start.is_month_start else 'ME')
    if freq_type == 'Quarterly': return pd.date_range(start, end, freq='QS' if start.is_quarter_start else 'QE')
    if freq_type == 'Yearly': return pd.date_range(start, end, freq='YS' if start.is_year_start else 'YE')
    return pd.DatetimeIndex([])
exp = expected_dates(df.Time.min(), df.Time.max(), frequency_type, inferred_freq)
missing = exp.difference(pd.DatetimeIndex(df.Time))
if len(missing):
    st.error(f'❌ พบช่วงเวลาขาดหาย {len(missing)} ช่วง กรุณาเติมหรือแก้ข้อมูลก่อนพยากรณ์')
    st.write(pd.DataFrame({'ช่วงเวลาที่ขาด': missing[:30]}))
    st.stop()

st.success(f'✅ ข้อมูลพร้อมวิเคราะห์ {len(df):,} ช่วงเวลา | Time column: {time_col} | Demand: {demand_col}')

# ---------- HELPERS ----------
def metrics(actual, pred):
    a,p = np.asarray(actual,float), np.asarray(pred,float)
    m = np.isfinite(a)&np.isfinite(p)
    a,p=a[m],p[m]
    if not len(a): return dict(MAPE=np.nan,MAD=np.nan,MSD=np.nan,RMSE=np.nan,N=0)
    e=a-p
    mad=float(np.mean(np.abs(e))); msd=float(np.mean(e**2)); rmse=float(np.sqrt(msd))
    nz=a!=0
    mape=float(np.mean(np.abs(e[nz]/a[nz]))*100) if np.any(nz) else np.nan
    return dict(MAPE=mape,MAD=mad,MSD=msd,RMSE=rmse,N=len(a))

def ma_one_step(y,w):
    # Use pandas rolling-window library function; shift(1) makes each value
    # a genuine one-step-ahead forecast using only observations available before t.
    return pd.Series(y, dtype='float64').rolling(window=w, min_periods=w).mean().shift(1).to_numpy()

def trend_one_step(y,min_history=3):
    out=np.full(len(y),np.nan)
    for i in range(min_history,len(y)):
        model=LinearRegression().fit(np.arange(i).reshape(-1,1),y[:i])
        out[i]=model.predict([[i]])[0]
    return out

def ses_one_step(y,min_history=3):
    out=np.full(len(y),np.nan)
    for i in range(min_history,len(y)):
        fit=SimpleExpSmoothing(y[:i],initialization_method='estimated').fit(optimized=True)
        out[i]=fit.forecast(1)[0]
    return out

def holt_one_step(y,min_history=4):
    # Holt จาก statsmodels โดยกำหนด alpha และ beta ให้อยู่ในช่วง 0-1
    out=np.full(len(y),np.nan)
    bounds={
        'smoothing_level': (0.0, 1.0),
        'smoothing_trend': (0.0, 1.0),
    }
    for i in range(min_history,len(y)):
        fit=ExponentialSmoothing(
            y[:i],
            trend='add',
            seasonal=None,
            initialization_method='estimated',
            bounds=bounds,
        ).fit(optimized=True)
        out[i]=fit.forecast(1)[0]
    return out

def seasonal_naive(y,p):
    out=np.full(len(y),np.nan)
    if p: out[p:]=y[:-p]
    return out

def fit_holt_winters_additive(y,p):
    # Holt-Winters Additive จาก statsmodels
    # บังคับ smoothing parameters alpha, beta และ gamma ให้อยู่ในช่วง 0-1
    if p is None:
        raise ValueError('ไม่พบ seasonal period')
    if len(y) < 2*p:
        raise ValueError(f'ต้องมีข้อมูลอย่างน้อย {2*p} ค่า สำหรับ Holt-Winters')

    bounds={
        'smoothing_level': (0.0, 1.0),
        'smoothing_trend': (0.0, 1.0),
        'smoothing_seasonal': (0.0, 1.0),
    }
    model=ExponentialSmoothing(
        np.asarray(y,float),
        trend='add',
        seasonal='add',
        seasonal_periods=int(p),
        initialization_method='estimated',
        bounds=bounds,
    )
    return model.fit(optimized=True)

def detect_trend(y):
    if len(y)<4:return False,np.nan,np.nan
    r=linregress(np.arange(len(y)),y)
    return bool(np.isfinite(r.pvalue) and r.pvalue<.05 and abs(r.slope)>1e-12),r.slope,r.pvalue

def detect_seasonality(y,p):
    if p is None or len(y)<2*p:
        return False,np.nan

    # Remove linear trend before checking seasonal-lag correlation.
    # This prevents a strong trend from being mistaken for seasonality.
    x=np.arange(len(y),dtype=float).reshape(-1,1)
    trend_model=LinearRegression().fit(x,y)
    detrended=y-trend_model.predict(x)

    a,b=detrended[p:],detrended[:-p]
    if np.std(a)==0 or np.std(b)==0:
        return False,np.nan
    r=float(np.corrcoef(a,b)[0,1])
    bound=1.96/np.sqrt(len(y))
    return bool(np.isfinite(r) and r>bound),r

def fmt_time(s,typ):
    s=pd.to_datetime(s)
    if typ in ('Daily','Weekly'): return s.dt.strftime('%Y-%m-%d')
    if typ=='Monthly': return s.dt.strftime('%Y-%m')
    if typ=='Quarterly': return s.dt.to_period('Q').astype(str)
    if typ=='Yearly': return s.dt.strftime('%Y')
    return s.astype(str)

def future_dates(last,n,typ,inferred):
    if inferred and classify_frequency(inferred)==typ:
        try:return pd.date_range(last,periods=n+1,freq=inferred)[1:]
        except Exception:pass
    offsets={'Daily':pd.DateOffset(days=1),'Weekly':pd.DateOffset(weeks=1),'Monthly':pd.DateOffset(months=1),'Quarterly':pd.DateOffset(months=3),'Yearly':pd.DateOffset(years=1)}
    off=offsets[typ]
    return pd.DatetimeIndex([last+off*i for i in range(1,n+1)])

# ---------- STRUCTURE / CANDIDATES ----------
y=df.Demand.to_numpy(float)
trend,slope,pval=detect_trend(y)
seasonal,season_r=detect_seasonality(y,seasonal_period)
st.header('🔎 ตรวจสอบลักษณะข้อมูล')
c1,c2,c3=st.columns(3)
c1.metric('Frequency',frequency_type)
c2.metric('Trend','✅' if trend else '❌')
c3.metric('Seasonality','✅' if seasonal else ('—' if seasonal_period is None else '❌'))

ma_max=min(12,max(2,len(y)//4))
ma_window=st.number_input('Moving Average length',min_value=2,max_value=ma_max,value=min(3,ma_max),step=1)

pred={}; status=[]
# MA and SES are candidates only for level-only series.
if not trend and not seasonal:
    pred[f'Moving Average ({ma_window}-period)']=ma_one_step(y,int(ma_window))
    try: pred['Simple Exponential Smoothing']=ses_one_step(y)
    except Exception as e: status.append(f'SES: {e}')
else:
    status.append('Moving Average / SES: ไม่ใช้เป็น candidate หลัก เพราะตรวจพบ Trend หรือ Seasonality')

if trend and not seasonal:
    pred['Trend Projection (Linear)']=trend_one_step(y)
    try: pred['Holt']=holt_one_step(y)
    except Exception as e: status.append(f'Holt: {e}')
else:
    status.append('Trend Projection / Holt: ใช้เป็น candidate หลักเมื่อมี Trend และไม่มี Seasonality')

if seasonal:
    if seasonal_period and len(y)>=2*seasonal_period:
        try:
            hw_fit = fit_holt_winters_additive(y, seasonal_period)
            pred['Holt-Winters Additive (Optimized)'] = np.asarray(hw_fit.fittedvalues,float)
        except Exception as e: status.append(f'Holt-Winters Additive: {e}')
    else:
        status.append('Holt-Winters Additive: ต้องมีข้อมูลอย่างน้อย 2 ฤดูกาล')

if not pred:
    # Safety fallback when detection is inconclusive.
    pred[f'Moving Average ({ma_window}-period)']=ma_one_step(y,int(ma_window))
    pred['Trend Projection (Linear)']=trend_one_step(y)
    status.append('ระบบไม่พบโครงสร้างชัดเจน จึงแสดงวิธีพื้นฐานเพื่อเปรียบเทียบ')

# ---------- MODEL ACCURACY + FAIR COMPARISON ----------
# 1) Accuracy ของแต่ละโมเดล: ใช้ทุก fitted / one-step-ahead value ที่โมเดลนั้นมี
#    ไม่บังคับ N ให้เท่ากัน เพื่อให้เห็น performance ของโมเดลนั้นเอง
model_rows=[]
for name,p in pred.items():
    own=np.isfinite(p) & np.isfinite(y)
    m=metrics(y[own],p[own])
    model_rows.append({
        'Model':name,'MAPE (%)':m['MAPE'],'MAD':m['MAD'],
        'MSD':m['MSD'],'RMSE':m['RMSE']
    })
model_acc=pd.DataFrame(model_rows)

st.header('📏 Accuracy ของแต่ละวิธี')
st.caption('แต่ละวิธีประเมินจาก fitted / one-step-ahead forecasts ที่วิธีนั้นสามารถคำนวณได้')
show_acc=model_acc.copy()
for c in ['MAPE (%)','MAD','MSD','RMSE']:
    show_acc[c]=show_acc[c].round(3)
st.dataframe(show_acc,width='stretch',hide_index=True)

# แสดง smoothing constants ของ Holt-Winters ที่ fit จาก statsmodels
if 'Holt-Winters Additive (Optimized)' in pred:
    try:
        a=hw_fit.params.get('smoothing_level',np.nan)
        b=hw_fit.params.get('smoothing_trend',np.nan)
        g=hw_fit.params.get('smoothing_seasonal',np.nan)
        st.caption(
            f'Holt-Winters Additive: α (Level) = {a:.4f} | '
            f'β (Trend) = {b:.4f} | γ (Seasonal) = {g:.4f}'
        )
    except Exception:
        pass

# 2) Model comparison: เมื่อมีมากกว่า 1 candidate จึงค่อยใช้ช่วง Actual เดียวกัน
#    เพื่อให้การจัดอันดับไม่เกิดจากโมเดลหนึ่งถูกวัดบนช่วงที่ง่ายกว่าอีกโมเดล
if len(pred) >= 2:
    valid=[np.isfinite(v) for v in pred.values()]
    common=np.logical_and.reduce(valid)

    if common.sum()>=2:
        rows=[]
        for name,p in pred.items():
            m=metrics(y[common],p[common])
            rows.append({
                'Model':name,'MAPE (%)':m['MAPE'],'MAD':m['MAD'],
                'MSD':m['MSD'],'RMSE':m['RMSE']
            })
        res=pd.DataFrame(rows)
        zero_in_eval=bool(np.any(y[common]==0))
        primary='MAD' if zero_in_eval else 'MAPE (%)'
        res=res.sort_values([primary,'RMSE'],na_position='last').reset_index(drop=True)
        res.insert(0,'Rank',np.arange(1,len(res)+1))
        best=res.iloc[0]['Model']

        # ไม่แสดงตารางเปรียบเทียบซ้ำกับ Accuracy ด้านบน
        # แต่ยังคงใช้ช่วง Actual เดียวกันภายในเพื่อเลือกโมเดลอย่างเป็นธรรม
        if zero_in_eval:
            st.warning('⚠️ ช่วงเปรียบเทียบมี Demand = 0 จึงไม่ใช้ MAPE เป็นเกณฑ์หลัก; ใช้ MAD แทน')
        st.success(
            f'วิธีที่มีค่าความคลาดเคลื่อนต่ำที่สุดตามเกณฑ์ **{primary}** '
            f'บนช่วงเปรียบเทียบเดียวกัน: **{best}**'
        )
    else:
        st.warning('⚠️ Candidate models มีช่วง fitted forecasts ร่วมกันน้อยกว่า 2 จุด จึงไม่จัดอันดับข้ามโมเดล')
        # ใช้โมเดลแรกเป็นค่าเริ่มต้นสำหรับส่วน Forecast เท่านั้น ไม่ประกาศว่าเป็น best model
        best=list(pred)[0]
else:
    best=list(pred)[0]
    st.info(f'มี candidate model ที่เหมาะสม 1 วิธี: **{best}** จึงไม่จำเป็นต้องจัดอันดับข้ามโมเดล')

with st.expander('ℹ️ เหตุผลการคัดเลือก/ไม่คัดเลือกวิธี'):
    for s in status:
        st.write('•',s)

# ---------- FUTURE ----------
st.divider(); st.header('🔮 พยากรณ์อนาคต')
future_model=st.selectbox('เลือกวิธีพยากรณ์',list(pred),index=list(pred).index(best))
hcfg={'Daily':(1,30,7,'วัน'),'Weekly':(1,12,4,'สัปดาห์'),'Monthly':(1,12,3,'เดือน'),'Quarterly':(1,8,4,'ไตรมาส'),'Yearly':(1,5,2,'ปี')}
lo,hi,de,unit=hcfg[frequency_type]
if future_model.startswith('Moving Average'):
    h=1; st.info(f'Moving Average แบบมาตรฐานพยากรณ์เฉพาะ 1 {unit}ถัดไป โดยไม่ใช้ forecast เป็น pseudo-actual')
else:
    h=st.slider(f'จำนวน{unit}ที่ต้องการพยากรณ์',lo,hi,de)

try:
    if future_model.startswith('Moving Average'):
        fc=np.array([pd.Series(y, dtype='float64').rolling(window=int(ma_window), min_periods=int(ma_window)).mean().iloc[-1]])
    elif future_model=='Simple Exponential Smoothing':
        fit=SimpleExpSmoothing(y,initialization_method='estimated').fit(optimized=True); fc=np.asarray(fit.forecast(h))
        st.caption(f"α (Level) = {fit.params.get('smoothing_level',np.nan):.4f}")
    elif future_model=='Trend Projection (Linear)':
        model=LinearRegression().fit(np.arange(len(y)).reshape(-1,1),y); fc=model.predict(np.arange(len(y),len(y)+h).reshape(-1,1))
    elif future_model=='Holt':
        fit=ExponentialSmoothing(
            y, trend='add', seasonal=None, initialization_method='estimated',
            bounds={'smoothing_level':(0.0,1.0),'smoothing_trend':(0.0,1.0)}
        ).fit(optimized=True); fc=np.asarray(fit.forecast(h))
        st.caption(f"α (Level) = {fit.params.get('smoothing_level',np.nan):.4f} | β (Trend) = {fit.params.get('smoothing_trend',np.nan):.4f}")
    elif future_model=='Holt-Winters Additive (Optimized)':
        fit=fit_holt_winters_additive(y,seasonal_period)
        fc=np.asarray(fit.forecast(h))
        a=fit.params.get('smoothing_level',np.nan)
        b=fit.params.get('smoothing_trend',np.nan)
        g=fit.params.get('smoothing_seasonal',np.nan)
        st.subheader('⚙️ Optimized Holt-Winters parameters')
        z1,z2,z3=st.columns(3)
        z1.metric('α (Level)',f'{a:.4f}')
        z2.metric('β (Trend)',f'{b:.4f}')
        z3.metric('γ (Seasonal)',f'{g:.4f}')
        st.caption('คำนวณด้วย statsmodels ExponentialSmoothing และกำหนด smoothing parameters ให้อยู่ในช่วง 0–1')
    else: raise ValueError('ไม่พบโมเดล')
except Exception as e:
    st.error(f'❌ ไม่สามารถพยากรณ์ได้: {e}'); st.stop()

fd=future_dates(df.Time.max(),h,frequency_type,inferred_freq)
out=pd.DataFrame({'Time':fd,'Forecast':np.round(fc,2)})
disp=out.copy(); disp['Time']=fmt_time(disp.Time,frequency_type)
st.subheader(f'ผลพยากรณ์โดย {future_model}')
st.dataframe(disp,width='stretch',hide_index=True)
# กราฟเดียว: Actual + fitted forecasts ของ candidate models + future forecast ของวิธีที่เลือก
hist_plot=pd.DataFrame({'Time':df.Time,'Actual':y})
for model_name, fitted_values in pred.items():
    hist_plot[model_name]=fitted_values

future_plot=out.rename(columns={'Forecast':f'{future_model} — Future Forecast'})
plot=pd.merge(hist_plot,future_plot,on='Time',how='outer').sort_values('Time').set_index('Time')

st.subheader('กราฟค่าจริงและค่าพยากรณ์')
st.caption('ช่วงข้อมูลย้อนหลังแสดง Actual และ fitted/one-step-ahead forecasts ของ Candidate Models; หลังข้อมูลจริงสิ้นสุดเป็น Future Forecast ของวิธีที่เลือก')
st.line_chart(plot)
st.download_button('⬇️ ดาวน์โหลดผลการพยากรณ์ (CSV)',disp.to_csv(index=False).encode('utf-8-sig'),'transportation_demand_forecast.csv','text/csv')

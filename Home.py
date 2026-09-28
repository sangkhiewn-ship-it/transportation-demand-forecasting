import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Transportation Demand Forecasting",
    page_icon="🚚",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {display: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🚚 Transportation Demand Forecasting")
st.subheader("การพยากรณ์ความต้องการการขนส่ง")
st.write("เว็บแอปพลิเคชันสำหรับวิเคราะห์ข้อมูลและพยากรณ์ความต้องการการขนส่ง")
st.divider()

st.header("📂 อัปโหลดข้อมูล")
st.write("เลือกไฟล์ข้อมูลความต้องการการขนส่ง")
st.caption(
    "📌 ในไฟล์ต้องมีคอลัมน์ Demand และคอลัมน์เวลาอย่างน้อย 1 คอลัมน์ ได้แก่ Date, Month หรือ Year "
    "(ชื่อคอลัมน์ใช้ตัวพิมพ์ใหญ่/เล็กได้ เช่น date, DATE, demand, DEMAND)"
)
st.caption("รูปแบบที่แนะนำ: Date = 2025-01-31 | Month = 2025-01 | Year = 2025")

uploaded_file = st.file_uploader(
    "อัปโหลดไฟล์",
    type=["csv", "xlsx"],
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.info("กรุณาอัปโหลดไฟล์ข้อมูล CSV หรือ Excel เพื่อเริ่มต้นการวิเคราะห์")

    st.markdown(
        """
        <div style="
            position: fixed;
            bottom: 10px;
            right: 15px;
            font-size: 12px;
            color: gray;
            text-align: right;
            z-index: 9999;
        ">
            Created by <b>Noppakun Sangkhiew</b>, Silpakorn University<br>
            Forecasting calculations use Python statistical libraries, including <b>statsmodels</b>.<br>
            <i>statsmodels: Seabold & Perktold (2010)</i>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.stop()

try:
    if uploaded_file.name.lower().endswith(".csv"):
        raw = pd.read_csv(uploaded_file)
    else:
        raw = pd.read_excel(uploaded_file)
except Exception as e:
    st.error("❌ ไม่สามารถอ่านไฟล์ได้")
    st.write("รายละเอียด:", str(e))
    st.stop()

# ---------------------------------------------------------
# Column validation: case-insensitive
# ---------------------------------------------------------
raw.columns = raw.columns.astype(str).str.strip()
colmap = {str(c).strip().lower(): c for c in raw.columns}

demand_col = colmap.get("demand")
time_candidates = [colmap[x] for x in ("date", "month", "year") if x in colmap]

if demand_col is None:
    st.error("❌ ไม่พบคอลัมน์ Demand กรุณาตรวจสอบไฟล์")
    st.info("ชื่อ Demand ใช้ตัวพิมพ์ใหญ่หรือเล็กได้ เช่น Demand, demand หรือ DEMAND")
    st.stop()

if not time_candidates:
    st.error("❌ ไม่พบคอลัมน์เวลา Date, Month หรือ Year")
    st.info("ชื่อคอลัมน์เวลาใช้ตัวพิมพ์ใหญ่หรือเล็กได้ เช่น Date/date/DATE, Month/month หรือ Year/year")
    st.stop()

if len(time_candidates) > 1:
    time_col = st.selectbox(
        "พบคอลัมน์เวลามากกว่า 1 คอลัมน์ กรุณาเลือกคอลัมน์ที่ต้องการใช้",
        time_candidates,
    )
else:
    time_col = time_candidates[0]

# ---------------------------------------------------------
# Parse Demand
# ---------------------------------------------------------
demand = pd.to_numeric(raw[demand_col], errors="coerce")
if demand.isna().any():
    st.error(
        f"❌ พบข้อมูลในคอลัมน์ {demand_col} ที่ไม่ใช่ตัวเลขหรือเป็นค่าว่าง "
        f"{int(demand.isna().sum())} รายการ"
    )
    st.stop()

# ---------------------------------------------------------
# Parse time. Column name does NOT determine frequency.
# Frequency is inferred from the actual sequence below.
# ---------------------------------------------------------
time_name = str(time_col).strip().lower()
source = raw[time_col]

# Support explicit quarter labels such as 2025-Q1.
quarter_text = source.astype(str).str.strip()
if quarter_text.str.match(r"^\d{4}-Q[1-4]$", case=False).all():
    try:
        parsed_time = pd.PeriodIndex(quarter_text.str.upper(), freq="Q").to_timestamp()
        parsed_time = pd.Series(parsed_time, index=raw.index)
    except Exception:
        parsed_time = pd.Series(pd.NaT, index=raw.index)
elif time_name == "year" and source.astype(str).str.strip().str.match(r"^\d{4}$").all():
    parsed_time = pd.to_datetime(
        source.astype(str).str.strip() + "-01-01",
        format="%Y-%m-%d",
        errors="coerce",
    )
else:
    parsed_time = pd.to_datetime(source, errors="coerce")

if parsed_time.isna().any():
    st.error(
        f"❌ พบข้อมูลในคอลัมน์เวลา {time_col} ที่ไม่สามารถอ่านได้ "
        f"{int(parsed_time.isna().sum())} รายการ"
    )
    st.info("รูปแบบที่แนะนำ: 2025-01-31, 2025-01, 2025-Q1 หรือ 2025")
    st.stop()

# Normalize internal data to Date + Demand so Home and Forecasting use the same structure.
df = pd.DataFrame({"Date": parsed_time, "Demand": demand.astype(float)})
df = df.sort_values("Date").reset_index(drop=True)

if len(df) < 2:
    st.error("❌ ข้อมูลมีจำนวนน้อยเกินไปสำหรับตรวจสอบความถี่")
    st.stop()

# ---------------------------------------------------------
# Duplicate time periods
# ---------------------------------------------------------
if df["Date"].duplicated().any():
    dup = df.loc[df["Date"].duplicated(keep=False), "Date"].drop_duplicates()
    st.error(f"❌ พบช่วงเวลาซ้ำ {len(dup)} ช่วงเวลา")
    st.write("ช่วงเวลาที่ซ้ำ:", ", ".join(d.strftime("%Y-%m-%d") for d in dup[:20]))
    st.stop()

# ---------------------------------------------------------
# Frequency detection from ACTUAL time spacing/order
# Do not classify YYYY-MM-DD as Daily merely because of its display format.
# ---------------------------------------------------------
def classify_frequency(freq):
    if not freq:
        return None
    f = str(freq).upper()
    if f == "D":
        return "Daily"
    if f.startswith("W") or f == "7D":
        return "Weekly"
    if f in ("MS", "M", "ME"):
        return "Monthly"
    if f.startswith(("Q", "QS", "QE")):
        return "Quarterly"
    if f.startswith(("Y", "A", "YS", "YE")):
        return "Yearly"
    return None

try:
    inferred_freq = pd.infer_freq(df["Date"]) if len(df) >= 3 else None
except Exception:
    inferred_freq = None

frequency_type = classify_frequency(inferred_freq)

# Fallback based on actual gaps if infer_freq cannot determine it.
if frequency_type is None:
    gaps = df["Date"].diff().dropna().dt.total_seconds() / 86400
    median_days = float(gaps.median()) if len(gaps) else None
    if median_days is not None:
        if 0.5 <= median_days <= 1.5:
            frequency_type = "Daily"
        elif 6 <= median_days <= 8:
            frequency_type = "Weekly"
        elif 27 <= median_days <= 32:
            frequency_type = "Monthly"
        elif 80 <= median_days <= 100:
            frequency_type = "Quarterly"
        elif 350 <= median_days <= 380:
            frequency_type = "Yearly"

frequency_labels = {
    "Daily": "รายวัน (Daily)",
    "Weekly": "รายสัปดาห์ (Weekly)",
    "Monthly": "รายเดือน (Monthly)",
    "Quarterly": "รายไตรมาส (Quarterly)",
    "Yearly": "รายปี (Yearly)",
}

if frequency_type is None:
    st.warning("⚠️ ระบบไม่สามารถระบุความถี่ของข้อมูลได้อย่างชัดเจน")
    selected_label = st.selectbox(
        "กรุณาเลือกความถี่ของข้อมูล",
        list(frequency_labels.values()),
    )
    frequency_type = next(k for k, v in frequency_labels.items() if v == selected_label)
else:
    st.info(f"ระบบตรวจพบความถี่: **{frequency_labels[frequency_type]}**")

# ---------------------------------------------------------
# Expected periods and missing-period check
# ---------------------------------------------------------
def expected_dates(start, end, freq_type, inferred=None):
    if inferred and classify_frequency(inferred) == freq_type:
        try:
            return pd.date_range(start, end, freq=inferred)
        except Exception:
            pass
    if freq_type == "Daily":
        return pd.date_range(start, end, freq="D")
    if freq_type == "Weekly":
        # Preserve the weekday represented by the first observation.
        return pd.date_range(start, end, freq=f"W-{start.day_name()[:3].upper()}")
    if freq_type == "Monthly":
        return pd.date_range(start, end, freq="MS" if start.is_month_start else "ME")
    if freq_type == "Quarterly":
        return pd.date_range(start, end, freq="QS" if start.is_quarter_start else "QE")
    if freq_type == "Yearly":
        return pd.date_range(start, end, freq="YS" if start.is_year_start else "YE")
    return pd.DatetimeIndex([])

expected = expected_dates(df["Date"].min(), df["Date"].max(), frequency_type, inferred_freq)
missing_dates = expected.difference(pd.DatetimeIndex(df["Date"]))

if len(missing_dates) > 0:
    st.error(f"❌ พบช่วงเวลาที่ขาดหาย {len(missing_dates)} ช่วงเวลา")
    if frequency_type == "Monthly":
        display = [d.strftime("%Y-%m") for d in missing_dates]
    elif frequency_type == "Quarterly":
        display = [f"{d.year}-Q{((d.month - 1) // 3) + 1}" for d in missing_dates]
    elif frequency_type == "Yearly":
        display = [d.strftime("%Y") for d in missing_dates]
    else:
        display = [d.strftime("%Y-%m-%d") for d in missing_dates]
    st.write("ช่วงเวลาที่ขาด:", ", ".join(display[:50]))
    if len(display) > 50:
        st.caption(f"แสดง 50 รายการแรก จากทั้งหมด {len(display)} รายการ")
    st.warning("ข้อมูล Time Series ควรมีช่วงเวลาต่อเนื่อง กรุณาตรวจสอบหรือเติมข้อมูลที่ขาดก่อนทำการพยากรณ์")
    st.stop()

# ---------------------------------------------------------
# Store normalized data for Forecasting page
# ---------------------------------------------------------
period_map = {"Daily": 7, "Weekly": 52, "Monthly": 12, "Quarterly": 4, "Yearly": None}
frequency_text = frequency_labels[frequency_type]
seasonal_period = period_map[frequency_type]

st.session_state["data"] = df.copy()
st.session_state["frequency_type"] = frequency_type
st.session_state["frequency_text"] = frequency_text
st.session_state["frequency_code"] = inferred_freq
st.session_state["seasonal_period"] = seasonal_period

st.success("✅ อัปโหลดและตรวจสอบข้อมูลสำเร็จ")
st.header("🔍 ข้อมูลเบื้องต้น")
st.write(f"จำนวนข้อมูล: {len(df)} ช่วงเวลา")
st.write(f"ความถี่ของข้อมูล: **{frequency_text}**")

start_date, end_date = df["Date"].min(), df["Date"].max()
if frequency_type == "Monthly":
    start_text, end_text = start_date.strftime("%Y-%m"), end_date.strftime("%Y-%m")
elif frequency_type == "Quarterly":
    start_text = f"{start_date.year}-Q{((start_date.month - 1) // 3) + 1}"
    end_text = f"{end_date.year}-Q{((end_date.month - 1) // 3) + 1}"
elif frequency_type == "Yearly":
    start_text, end_text = start_date.strftime("%Y"), end_date.strftime("%Y")
else:
    start_text, end_text = start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
st.write(f"ช่วงข้อมูล: {start_text} – {end_text}")
st.write("Missing Demand: 0")
st.write("ช่วงเวลาซ้ำ: 0")
st.write("ช่วงเวลาที่ขาด: 0")


st.markdown(
    """
    <div style="
        position: fixed;
        bottom: 10px;
        right: 15px;
        font-size: 12px;
        color: gray;
        text-align: right;
        z-index: 9999;
    ">
        Created by <b>Noppakun Sangkhiew</b>, Silpakorn University<br>
        Forecasting calculations use Python statistical libraries, including <b>statsmodels</b>.<br>
        <i>statsmodels: Seabold & Perktold (2010)</i>
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()
if st.button("🔮 ทำการพยากรณ์ข้อมูล", type="primary", use_container_width=True):
    st.switch_page("pages/1_Forecasting.py")

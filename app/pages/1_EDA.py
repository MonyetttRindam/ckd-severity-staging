import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

# ------------------------------------------------------------
# Konfigurasi halaman
# ------------------------------------------------------------
st.set_page_config(page_title="EDA – CKD Dataset", layout="wide", page_icon="📊")

# ------------------------------------------------------------
# Custom CSS (konsisten dengan halaman utama)
# ------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #F0F4F8;
}

/* Hero */
.eda-hero {
    background: linear-gradient(135deg, #1B3A6B 0%, #1A7A8A 100%);
    border-radius: 16px;
    padding: 28px 36px 24px 36px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px rgba(27, 58, 107, 0.18);
}
.eda-hero h1 {
    color: #FFFFFF;
    font-size: 1.6rem;
    font-weight: 700;
    margin: 0 0 6px 0;
    letter-spacing: -0.3px;
}
.eda-hero p {
    color: #B8DCEB;
    font-size: 0.9rem;
    margin: 0;
    line-height: 1.6;
}

/* Stat badge row */
.stat-row {
    display: flex;
    gap: 14px;
    margin-bottom: 22px;
    flex-wrap: wrap;
}
.stat-badge {
    background: #FFFFFF;
    border: 1px solid #DDE4EE;
    border-radius: 10px;
    padding: 14px 22px;
    flex: 1;
    min-width: 140px;
    box-shadow: 0 2px 6px rgba(27,58,107,0.06);
}
.stat-badge .val {
    font-size: 1.5rem;
    font-weight: 700;
    color: #1B3A6B;
    line-height: 1;
}
.stat-badge .lbl {
    font-size: 0.75rem;
    color: #6B7A90;
    margin-top: 4px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* Chart card */
.chart-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 22px 24px;
    border: 1px solid #DDE4EE;
    box-shadow: 0 2px 8px rgba(27, 58, 107, 0.06);
    margin-bottom: 20px;
}

/* Section label */
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: #1A7A8A;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 2px solid #E3EDF5;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #1B3A6B !important;
}
section[data-testid="stSidebar"] * {
    color: #E8EFF8 !important;
}
/* Opsi di dalam dropdown selectbox sidebar */
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div {
    background-color: #1B3A6B !important;
    color: #FFFFFF !important;
}

section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="popover"] li {
    background-color: #1B3A6B !important;
    color: #E8EFF8 !important;
}

section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="popover"] li:hover {
    background-color: #1A7A8A !important;
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: #B8DCEB !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 16px !important;
}
section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
}

/* Dataframe */
.stDataFrame {
    border-radius: 10px !important;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("CKD_NHANES_2021_2023.csv")
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ File 'CKD_NHANES_2021_2023.csv' tidak ditemukan. Pastikan ada di root repositori.")
    st.stop()

num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
all_cols = df.columns.tolist()

# Cek apakah ada kolom target CKD
target_col = None
for c in ["ckd_label", "CKD", "ckd", "label", "target"]:
    if c in df.columns:
        target_col = c
        break

# ------------------------------------------------------------
# Hero
# ------------------------------------------------------------
st.markdown(f"""
<div class="eda-hero">
    <h1>📊 Eksplorasi Dataset CKD</h1>
    <p>Visualisasi dataset <strong>mentah (sebelum preprocessing)</strong> — gunakan panel kiri untuk memilih jenis plot dan fitur.</p>
</div>
""", unsafe_allow_html=True)

# Stat badges
missing_total = df.isnull().sum().sum()
num_count = len(num_cols)
cat_count = len(cat_cols)
st.markdown(f"""
<div class="stat-row">
    <div class="stat-badge"><div class="val">{df.shape[0]:,}</div><div class="lbl">Total Baris</div></div>
    <div class="stat-badge"><div class="val">{df.shape[1]}</div><div class="lbl">Total Kolom</div></div>
    <div class="stat-badge"><div class="val">{num_count}</div><div class="lbl">Fitur Numerik</div></div>
    <div class="stat-badge"><div class="val">{cat_count}</div><div class="lbl">Fitur Kategorikal</div></div>
    <div class="stat-badge"><div class="val">{missing_total:,}</div><div class="lbl">Missing Values</div></div>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.sidebar.markdown("### 📊 Jenis Visualisasi")
plot_type = st.sidebar.selectbox(
    "Pilih plot",
    ["📊 Bar Count", "📈 Histogram", "📦 Box Plot",
     "🥧 Pie Chart", "🔵 Scatter Plot",
     "🔥 Correlation Heatmap", "📐 Pair Plot",
     "📋 Summary Statistics"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Opsi Plot")

# Plotly color palette konsisten
PALETTE = px.colors.sequential.Teal
BAR_COLOR = "#1A7A8A"
CHART_TEMPLATE = "plotly_white"

def styled_chart(fig):
    """Apply consistent styling to all plotly figures."""
    fig.update_layout(
        template=CHART_TEMPLATE,
        font_family="Inter, sans-serif",
        title_font=dict(size=15, color="#1B3A6B", family="Inter, sans-serif"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=30, l=10, r=10),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#DDE4EE",
            borderwidth=1,
            font=dict(size=11)
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EDF2F7", gridwidth=1, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#EDF2F7", gridwidth=1, zeroline=False)
    return fig

# ============================================================
# 1. BAR COUNT
# ============================================================
if plot_type == "📊 Bar Count":
    if not cat_cols:
        st.warning("Tidak ada kolom kategorikal.")
    else:
        col = st.sidebar.selectbox("Kolom kategorikal", cat_cols)
        sort_by = st.sidebar.radio("Urutkan berdasarkan", ["Frekuensi (↓)", "Alfabet"])
        show_pct = st.sidebar.checkbox("Tampilkan persentase", value=True)
        color_by_target = False
        if target_col:
            color_by_target = st.sidebar.checkbox(f"Warna per '{target_col}'", value=False)

        freq = df[col].value_counts().reset_index()
        freq.columns = ["Kategori", "Frekuensi"]
        freq["Persentase (%)"] = (freq["Frekuensi"] / freq["Frekuensi"].sum() * 100).round(1)

        if sort_by == "Alfabet":
            freq = freq.sort_values("Kategori")

        text_col = "Persentase (%)" if show_pct else "Frekuensi"
        text_fmt = [f"{v}%" for v in freq["Persentase (%)"]] if show_pct else freq["Frekuensi"].tolist()

        if color_by_target and target_col:
            # grouped bar
            grouped = df.groupby([col, target_col]).size().reset_index(name="Frekuensi")
            fig = px.bar(grouped, x=col, y="Frekuensi", color=str(target_col),
                         barmode="group", title=f"Distribusi {col} per {target_col}",
                         color_discrete_sequence=["#1B3A6B", "#1A7A8A"])
        else:
            fig = px.bar(freq, x="Kategori", y="Frekuensi",
                         title=f"Distribusi {col}",
                         text=[str(t) for t in text_fmt],
                         color_discrete_sequence=[BAR_COLOR])
            fig.update_traces(textposition="outside", marker_line_width=0)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(styled_chart(fig), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 2. HISTOGRAM
# ============================================================
elif plot_type == "📈 Histogram":
    if not num_cols:
        st.warning("Tidak ada kolom numerik.")
    else:
        col = st.sidebar.selectbox("Kolom numerik", num_cols)
        bins = st.sidebar.slider("Jumlah bins", 5, 100, 30)
        show_kde = st.sidebar.checkbox("Overlay kurva KDE", value=True)
        show_rug = st.sidebar.checkbox("Tampilkan rug plot", value=False)
        if target_col:
            split_by_target = st.sidebar.checkbox(f"Pisah per '{target_col}'", value=False)
        else:
            split_by_target = False

        col_data = df[col].dropna()

        if split_by_target and target_col:
            fig = px.histogram(df, x=col, color=str(target_col), nbins=bins,
                               title=f"Distribusi {col} per {target_col}",
                               marginal="rug" if show_rug else "box",
                               barmode="overlay", opacity=0.7,
                               color_discrete_sequence=["#1B3A6B", "#1A7A8A"])
        else:
            marginal = "rug" if show_rug else "box"
            fig = px.histogram(df, x=col, nbins=bins,
                               title=f"Distribusi {col}",
                               marginal=marginal, opacity=0.85,
                               color_discrete_sequence=[BAR_COLOR])

        if show_kde and not split_by_target:
            col_clean = col_data.values
            kde = stats.gaussian_kde(col_clean)
            x_range = np.linspace(col_clean.min(), col_clean.max(), 300)
            # scale KDE to match histogram counts
            bin_width = (col_clean.max() - col_clean.min()) / bins
            kde_y = kde(x_range) * len(col_clean) * bin_width
            fig.add_trace(go.Scatter(
                x=x_range, y=kde_y,
                mode="lines", name="KDE",
                line=dict(color="#E85D04", width=2.5)
            ))

        # Anotasi statistik
        mean_val = col_data.mean()
        median_val = col_data.median()
        fig.add_vline(x=mean_val, line_dash="dash", line_color="#1B3A6B",
                      annotation_text=f"Mean: {mean_val:.2f}", annotation_position="top right")
        fig.add_vline(x=median_val, line_dash="dot", line_color="#1A7A8A",
                      annotation_text=f"Median: {median_val:.2f}", annotation_position="top left")

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(styled_chart(fig), use_container_width=True)

        # Stats ringkas
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{mean_val:.2f}")
        c2.metric("Median", f"{median_val:.2f}")
        c3.metric("Std Dev", f"{col_data.std():.2f}")
        c4.metric("Skewness", f"{col_data.skew():.2f}")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 3. BOX PLOT
# ============================================================
elif plot_type == "📦 Box Plot":
    if not num_cols:
        st.warning("Tidak ada kolom numerik.")
    else:
        col = st.sidebar.selectbox("Kolom numerik", num_cols)
        group_col = st.sidebar.selectbox("Kelompokkan per kolom (opsional)", ["—"] + cat_cols)
        show_points = st.sidebar.radio("Tampilkan data points", ["Outlier saja", "Semua", "Tidak"])
        points_map = {"Outlier saja": "outliers", "Semua": "all", "Tidak": False}

        g = None if group_col == "—" else group_col
        fig = px.box(df, y=col, x=g, color=g,
                     title=f"Box Plot: {col}" + (f" per {g}" if g else ""),
                     points=points_map[show_points],
                     color_discrete_sequence=px.colors.sequential.Teal[2:])
        fig.update_traces(boxmean="sd")  # tampilkan mean & SD di dalam box

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(styled_chart(fig), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 4. PIE CHART
# ============================================================
elif plot_type == "🥧 Pie Chart":
    if not cat_cols:
        st.warning("Tidak ada kolom kategorikal.")
    else:
        col = st.sidebar.selectbox("Kolom kategorikal", cat_cols)
        top_n = st.sidebar.slider("Jumlah kategori teratas", 3, 15, 8)
        donut = st.sidebar.checkbox("Mode Donut", value=True)

        top_cats = df[col].value_counts().nlargest(top_n).index
        df_f = df[df[col].isin(top_cats)]
        fig = px.pie(df_f, names=col,
                     title=f"Proporsi {col} (Top {top_n})",
                     color_discrete_sequence=px.colors.sequential.Teal,
                     hole=0.4 if donut else 0)
        fig.update_traces(textinfo="label+percent", pull=[0.03] * top_n)

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(styled_chart(fig), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 5. SCATTER PLOT
# ============================================================
elif plot_type == "🔵 Scatter Plot":
    if len(num_cols) < 2:
        st.warning("Butuh minimal 2 kolom numerik.")
    else:
        x_col = st.sidebar.selectbox("Sumbu X", num_cols, index=0)
        y_col = st.sidebar.selectbox("Sumbu Y", num_cols, index=min(1, len(num_cols)-1))
        color_col = st.sidebar.selectbox("Warna per kolom (opsional)", ["—"] + cat_cols + num_cols)
        size_col = st.sidebar.selectbox("Ukuran titik (opsional)", ["—"] + num_cols)
        trendline = st.sidebar.selectbox("Trendline", ["Tidak", "OLS (Linear)", "LOWESS"])

        tl_map = {"Tidak": None, "OLS (Linear)": "ols", "LOWESS": "lowess"}
        color_arg = None if color_col == "—" else color_col
        size_arg = None if size_col == "—" else size_col

        fig = px.scatter(
            df, x=x_col, y=y_col,
            color=color_arg, size=size_arg,
            title=f"Scatter: {x_col} vs {y_col}",
            trendline=tl_map[trendline],
            trendline_color_override="#E85D04",
            hover_data={c: True for c in all_cols[:8]},
            opacity=0.65,
            color_discrete_sequence=px.colors.sequential.Teal,
            color_continuous_scale="teal"
        )

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(styled_chart(fig), use_container_width=True)

        # Korelasi cepat
        if x_col != y_col:
            r, p = stats.pearsonr(df[[x_col, y_col]].dropna()[x_col],
                                  df[[x_col, y_col]].dropna()[y_col])
            direction = "positif" if r > 0 else "negatif"
            strength = "kuat" if abs(r) > 0.6 else ("sedang" if abs(r) > 0.3 else "lemah")
            st.info(f"**Korelasi Pearson:** r = {r:.3f} (p = {p:.4f}) — korelasi {strength} {direction}")
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 6. CORRELATION HEATMAP
# ============================================================
elif plot_type == "🔥 Correlation Heatmap":
    if len(num_cols) < 2:
        st.warning("Butuh minimal 2 kolom numerik.")
    else:
        selected_cols = st.sidebar.multiselect(
            "Pilih fitur (kosong = semua numerik)",
            num_cols, default=[]
        )
        method = st.sidebar.radio("Metode korelasi", ["Pearson", "Spearman", "Kendall"])
        show_vals = st.sidebar.checkbox("Tampilkan nilai korelasi", value=True)

        use_cols = selected_cols if selected_cols else num_cols
        corr = df[use_cols].corr(method=method.lower())

        fig = px.imshow(
            corr,
            text_auto=".2f" if show_vals else False,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title=f"Heatmap Korelasi ({method}) — {len(use_cols)} Fitur"
        )
        fig.update_traces(textfont=dict(size=10))
        fig.update_coloraxes(colorbar=dict(
            title="r", tickvals=[-1, -0.5, 0, 0.5, 1]
        ))

        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(styled_chart(fig), use_container_width=True)

        # Top 5 korelasi terkuat (absolut, exclude diagonal)
        corr_flat = corr.abs().unstack()
        corr_flat = corr_flat[corr_flat < 1].sort_values(ascending=False)
        seen = set()
        top_pairs = []
        for (a, b), v in corr_flat.items():
            pair = frozenset([a, b])
            if pair not in seen:
                seen.add(pair)
                top_pairs.append((a, b, round(corr.loc[a, b], 3)))
            if len(top_pairs) == 5:
                break

        st.markdown('<div class="section-label" style="margin-top:12px;">🔝 5 Pasangan Fitur dengan Korelasi Terkuat</div>', unsafe_allow_html=True)
        top_df = pd.DataFrame(top_pairs, columns=["Fitur A", "Fitur B", f"r ({method})"])
        st.dataframe(top_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 7. PAIR PLOT
# ============================================================
elif plot_type == "📐 Pair Plot":
    if len(num_cols) < 2:
        st.warning("Butuh minimal 2 kolom numerik.")
    else:
        selected = st.sidebar.multiselect(
            "Pilih fitur numerik (max 6 untuk performa)",
            num_cols, default=num_cols[:4]
        )
        if target_col:
            use_color = st.sidebar.checkbox(f"Warnai per '{target_col}'", value=True)
        else:
            use_color = False

        if len(selected) < 2:
            st.warning("Pilih minimal 2 fitur.")
        else:
            color_arg = str(target_col) if (use_color and target_col) else None
            fig = px.scatter_matrix(
                df, dimensions=selected,
                color=color_arg,
                title="Pair Plot (Scatter Matrix)",
                color_discrete_sequence=["#1B3A6B", "#1A7A8A"],
                opacity=0.5
            )
            fig.update_traces(diagonal_visible=True, showupperhalf=False,
                              marker=dict(size=3))
            fig.update_layout(height=600)

            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(styled_chart(fig), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 8. SUMMARY STATISTICS
# ============================================================
elif plot_type == "📋 Summary Statistics":
    tab1, tab2, tab3 = st.tabs(["📊 Statistik Deskriptif", "❓ Missing Values", "🔍 Preview Data"])

    with tab1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Statistik Deskriptif (Numerik & Kategorikal)</div>', unsafe_allow_html=True)
        st.dataframe(df.describe(include="all").T.style.format(precision=2), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        missing = df.isnull().sum().reset_index()
        missing.columns = ["Kolom", "Jumlah Missing"]
        missing["Persen (%)"] = (missing["Jumlah Missing"] / len(df) * 100).round(2)
        missing = missing[missing["Jumlah Missing"] > 0].sort_values("Jumlah Missing", ascending=False)

        if missing.empty:
            st.success("✅ Tidak ada missing values di dataset ini.")
        else:
            fig = px.bar(
                missing, x="Kolom", y="Persen (%)",
                text="Persen (%)",
                title=f"Missing Values per Kolom ({len(missing)} kolom terpengaruh)",
                color="Persen (%)",
                color_continuous_scale="Blues",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", marker_line_width=0)
            fig.update_layout(showlegend=False, coloraxis_showscale=False)
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.plotly_chart(styled_chart(fig), use_container_width=True)
            st.dataframe(missing, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        n_rows = st.slider("Jumlah baris yang ditampilkan", 10, 200, 50)
        st.dataframe(df.head(n_rows), use_container_width=True)

# ------------------------------------------------------------
# Sidebar footer
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<p style="font-size:0.75rem; color:#7A9DC0; text-align:center;">CKD Early Screening · EDA Module</p>',
    unsafe_allow_html=True
)

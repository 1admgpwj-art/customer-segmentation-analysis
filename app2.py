
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

st.set_page_config(page_title="顧客セグメント分析ダッシュボード", layout="wide")

st.title("顧客セグメント分析ダッシュボード")
st.caption("RFM分析×K-meansクラスタリングによる顧客グループ分析")

@st.cache_data
def load_and_analyze():
    df = pd.read_csv("Sample - Superstore.csv", encoding="latin-1")
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    
    reference_date = df["Order Date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer ID").agg(
        Recency=("Order Date", lambda x: (reference_date - x.max()).days),
        Frequency=("Order ID", "nunique"),
        Monetary=("Sales", "sum")
    ).reset_index()
    
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency","Frequency","Monetary"]])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
    
    cluster_names = {0: "離脱リスク顧客", 1: "優良顧客", 2: "一般顧客"}
    rfm["顧客グループ"] = rfm["Cluster"].map(cluster_names)
    return df, rfm

df, rfm = load_and_analyze()

# KPIカード
st.subheader("顧客グループ概要")
col1, col2, col3, col4 = st.columns(4)
col1.metric("総顧客数", f"{len(rfm):,}人")
col2.metric("優良顧客", 
            f"{len(rfm[rfm['顧客グループ']=='優良顧客']):,}人",
            f"{len(rfm[rfm['顧客グループ']=='優良顧客'])/len(rfm)*100:.1f}%")
col3.metric("一般顧客",
            f"{len(rfm[rfm['顧客グループ']=='一般顧客']):,}人",
            f"{len(rfm[rfm['顧客グループ']=='一般顧客'])/len(rfm)*100:.1f}%")
col4.metric("離脱リスク顧客",
            f"{len(rfm[rfm['顧客グループ']=='離脱リスク顧客']):,}人",
            f"-{len(rfm[rfm['顧客グループ']=='離脱リスク顧客'])/len(rfm)*100:.1f}%")

st.divider()

# 構成比と売上貢献
st.subheader("顧客グループの構成と売上貢献")
col_l, col_r = st.columns(2)

with col_l:
    group_counts = rfm["顧客グループ"].value_counts()
    colors = {"優良顧客": "#4CAF50", "一般顧客": "#2196F3", "離脱リスク顧客": "#F44336"}
    fig1 = px.pie(values=group_counts.values, names=group_counts.index,
                  title="顧客グループの構成比",
                  color=group_counts.index,
                  color_discrete_map=colors, hole=0.4)
    fig1.update_traces(textinfo="percent+label+value")
    st.plotly_chart(fig1, use_container_width=True)

with col_r:
    group_rev = rfm.groupby("顧客グループ")["Monetary"].sum().reset_index()
    fig2 = px.pie(group_rev, values="Monetary", names="顧客グループ",
                  title="顧客グループ別 売上貢献",
                  color="顧客グループ",
                  color_discrete_map=colors, hole=0.4)
    fig2.update_traces(textinfo="percent+label")
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# RFM指標比較
st.subheader("顧客グループ別 RFM指標比較")
cluster_summary = rfm.groupby("顧客グループ").agg(
    平均Recency=("Recency","mean"),
    平均Frequency=("Frequency","mean"),
    平均Monetary=("Monetary","mean")
).round(1).reset_index()

fig3 = make_subplots(rows=1, cols=3,
                     subplot_titles=("Recency 低いほど良い",
                                    "Frequency 高いほど良い", 
                                    "Monetary 高いほど良い"))
for i, metric in enumerate(["平均Recency","平均Frequency","平均Monetary"]):
    for _, row in cluster_summary.iterrows():
        fig3.add_trace(
            go.Bar(x=[row["顧客グループ"]], y=[row[metric]],
                   name=row["顧客グループ"],
                   marker_color=colors[row["顧客グループ"]],
                   showlegend=(i==0),
                   hovertemplate=f"%{{x}}<br>{metric}: %{{y:.1f}}<extra></extra>"),
            row=1, col=i+1)
fig3.update_layout(plot_bgcolor="white", height=400)
fig3.update_yaxes(gridcolor="#eeeeee")
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ビジネス提言
st.subheader("ビジネス提言")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.success("優良顧客（34%）への施策")
    st.write("- VIPプログラムの導入")
    st.write("- 限定特典・先行案内")
    st.write("- 目的：離脱防止が最優先")

with col_b:
    st.info("一般顧客（52%）への施策")
    st.write("- 購入後フォローメール")
    st.write("- リピート促進キャンペーン")
    st.write("- 目的：最大の成長機会")

with col_c:
    st.error("離脱リスク顧客（14%）への施策")
    st.write("- 復帰キャンペーン")
    st.write("- 原因把握アンケート")
    st.write("- 目的：予防と現状把握")

st.divider()
st.caption("※ビジネスインパクト試算は単純試算です。実際の施策効果は転換率・コスト等を考慮した詳細分析が必要です。")

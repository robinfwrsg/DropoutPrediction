import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.stats as stats

# Load datasets
student_df = pd.read_csv("Student_Records.csv")
school_df = pd.read_csv("School_Info.csv")
teacher_df = pd.read_csv("Teacher_Deployment.csv")

# Merge student and school data
student_school_df = pd.merge(student_df, school_df, on='School_ID', how='left')
student_school_df['Attendance_Rate'] = pd.to_numeric(student_school_df['Attendance_Rate'], errors='coerce')

# Aggregate teacher experience by School_ID
teacher_exp_df = teacher_df.groupby('School_ID')['Years_of_Experience'].mean().reset_index()
teacher_exp_df.rename(columns={'Years_of_Experience': 'Avg_Teacher_Experience'}, inplace=True)

# Merge teacher experience info into student-school df
full_df = pd.merge(student_school_df, teacher_exp_df, on='School_ID', how='left')

# --- KPIs ---
total_students = len(full_df)
dropout_count = (full_df['Dropout_Status'] == 'Y').sum()
overall_dropout_rate = dropout_count / total_students * 100
avg_attendance = full_df['Attendance_Rate'].mean()
avg_teacher_exp = full_df['Avg_Teacher_Experience'].mean()
avg_infra = full_df['Infrastructure_Score'].mean()

st.set_page_config(page_title="Dropout Prediction Dashboard", layout="wide")
st.title("Dropout Prediction KPIs Dashboard")
st.markdown("Professional insights into dropout and related factors.")

# KPI cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Overall Dropout Rate", f"{overall_dropout_rate:.2f}%")
kpi2.metric("Average Attendance Rate", f"{avg_attendance:.2f}%")
kpi3.metric("Average Teacher Experience (years)", f"{avg_teacher_exp:.2f}")
kpi4.metric("Average Infrastructure Score", f"{avg_infra:.2f} / 5")

st.markdown("---")

# --- 1) Overall Dropout Rate Pie Chart ---
dropout_pie = go.Figure(go.Pie(
    labels=['Dropped Out', 'Not Dropped Out'],
    values=[dropout_count, total_students - dropout_count],
    hole=0.5,
    marker_colors=['#636EFA', '#B0BEC5'],
    hoverinfo='label+percent',
    textinfo='percent'
))
dropout_pie.update_layout(margin=dict(t=0,b=0,l=0,r=0), font=dict(family='Arial'), showlegend=True)

# --- 2) Infrastructure Score vs Dropout Line with 95% CI ---
infra_stats = full_df.groupby('Dropout_Status')['Infrastructure_Score'].agg(['mean', 'count', 'std'])
infra_stats['ci95'] = 1.96 * infra_stats['std'] / np.sqrt(infra_stats['count'])

infra_line = go.Figure()
infra_line.add_trace(go.Scatter(
    x=infra_stats.index,
    y=infra_stats['mean'],
    mode='lines+markers',
    line=dict(color='#1976D2', width=3),
    error_y=dict(type='data', array=infra_stats['ci95'], visible=True, color='#1976D2'),
    name='Mean Infrastructure Score'
))
infra_line.update_layout(
    margin=dict(t=20,b=30,l=40,r=20),
    yaxis_title="Infrastructure Score",
    xaxis_title="Dropout Status",
    yaxis=dict(range=[1, 5]),
    font=dict(family='Arial'),
    showlegend=False
)

# --- 3) Attendance Rate Distribution Box Plot by Dropout Status ---
att_dropout_df = full_df[['Attendance_Rate', 'Dropout_Status']].dropna()

# Statistical test
att_group_yes = att_dropout_df.loc[att_dropout_df['Dropout_Status'] == 'Y', 'Attendance_Rate']
att_group_no = att_dropout_df.loc[att_dropout_df['Dropout_Status'] == 'N', 'Attendance_Rate']
_, p_val_att = stats.mannwhitneyu(att_group_yes, att_group_no, alternative='two-sided')

attendance_box = go.Figure()
attendance_box.add_trace(go.Box(
    y=att_group_no,
    name='No Dropout',
    marker_color='#1E88E5',
    boxmean='sd'
))
attendance_box.add_trace(go.Box(
    y=att_group_yes,
    name='Dropped Out',
    marker_color='#D32F2F',
    boxmean='sd'
))
attendance_box.update_layout(
    margin=dict(t=30,b=40,l=40,r=20),
    yaxis_title="Attendance Rate (%)",
    # Simplified title to avoid overlap:
    title=f"Attendance Rate by Dropout Status\n(p={p_val_att:.3f})",
    font=dict(family='Arial'),
    showlegend=False,
    plot_bgcolor='white',
    yaxis=dict(showgrid=True, gridcolor='#E5ECF6')
)

# --- 4) Teacher Experience Distribution Box Plot by Dropout Status ---
teacher_dropout_df = full_df[['Avg_Teacher_Experience', 'Dropout_Status']].dropna()

# Statistical test
teach_group_yes = teacher_dropout_df.loc[teacher_dropout_df['Dropout_Status'] == 'Y', 'Avg_Teacher_Experience']
teach_group_no = teacher_dropout_df.loc[teacher_dropout_df['Dropout_Status'] == 'N', 'Avg_Teacher_Experience']
_, p_val_teach = stats.mannwhitneyu(teach_group_yes, teach_group_no, alternative='two-sided')

teacher_box = go.Figure()
teacher_box.add_trace(go.Box(
    y=teach_group_no,
    name='No Dropout',
    marker_color='#3949AB',
    boxmean='sd'
))
teacher_box.add_trace(go.Box(
    y=teach_group_yes,
    name='Dropped Out',
    marker_color='#D32F2F',
    boxmean='sd'
))
teacher_box.update_layout(
    margin=dict(t=30,b=40,l=40,r=20),
    yaxis_title="Avg Teacher Experience (years)",
    title=f"Teacher Experience by Dropout Status\n(p={p_val_teach:.3f})",
    font=dict(family='Arial'),
    showlegend=False,
    plot_bgcolor='white',
    yaxis=dict(showgrid=True, gridcolor='#E5ECF6')
)

# --- 5) New KPI: Dropout Rate by District Bar Chart ---
district_dropout = full_df.groupby('District')['Dropout_Status'].apply(lambda x: (x=='Y').mean()).reset_index()
district_dropout['Dropout_Rate'] = district_dropout['Dropout_Status'] * 100
district_dropout = district_dropout.sort_values('Dropout_Rate', ascending=False)

district_bar = go.Figure(go.Bar(
    x=district_dropout['District'],
    y=district_dropout['Dropout_Rate'],
    marker_color='#1976D2'
))
district_bar.update_layout(
    margin=dict(t=20,b=40,l=40,r=20),
    yaxis_title="Dropout Rate (%)",
    xaxis_title="District",
    font=dict(family='Arial'),
    plot_bgcolor='white',
    yaxis=dict(showgrid=True, gridcolor='#E5ECF6'),
    # Remove title to prevent overlap:
    title=None
)

# --- Layout ---
row1_col1, row1_col2, row1_col3 = st.columns(3)
row2_col1, row2_col2 = st.columns(2)

with row1_col1:
    st.markdown("### Overall Dropout Rate")
    st.plotly_chart(dropout_pie, use_container_width=True)

with row1_col2:
    st.markdown("### Infrastructure Score by Dropout Status")
    st.plotly_chart(infra_line, use_container_width=True)

with row1_col3:
    st.markdown("### Attendance Rate by Dropout Status")
    st.plotly_chart(attendance_box, use_container_width=True)

with row2_col1:
    st.markdown("### Teacher Experience by Dropout Status")
    st.plotly_chart(teacher_box, use_container_width=True)

with row2_col2:
    st.markdown("### Dropout Rate by District")
    st.plotly_chart(district_bar, use_container_width=True)

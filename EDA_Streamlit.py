import streamlit as st
import pandas as pd
import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp



name = "Credit Card Customers EDA Dashboard"

st.header(name)


dataset_url = "https://raw.githubusercontent.com/ScarlettQiu/credit_card_customer_churn_prediction/af46862bff782f1008725fa49db88d1e36fc8d43/BankChurners.csv"

@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url)

df = get_data()

with st.sidebar:
    st.write("Average Figures of Credit Card Customers")
    job_filter = st.selectbox("Select Customer Status", pd.unique(df["Attrition_Flag"]))

# top-level filters
#job_filter = st.selectbox("Select Customer Status", pd.unique(df["Attrition_Flag"]))

# dataframe filter
    df = df[df["Attrition_Flag"] == job_filter]

    avg_age = df['Customer_Age'].mean()
    avg_dep = df['Dependent_count'].mean()
    count_married = df['Marital_Status'][df['Marital_Status'] == 'Married'].count()
    count = df['Marital_Status'].count()
    percent_married = round(count_married/count, 2)

    avg_trans = df['Total_Trans_Ct'].mean()
    avg_Amt = df['Total_Trans_Amt'].mean()
    avg_amt_ch = df['Total_Amt_Chng_Q4_Q1'].mean()
    avg_ct_ch = df['Total_Ct_Chng_Q4_Q1'].mean()
    avg_relation = df['Months_on_book'].mean()
    num_prod = df['Total_Relationship_Count'].mean()
    mon_inactive = df['Months_Inactive_12_mon'].mean()
    contact_mon = df['Contacts_Count_12_mon'].mean()
    credit_lim = df['Credit_Limit'].mean()
    revolving = df['Total_Revolving_Bal'].mean()
    open_to_buy = df['Avg_Open_To_Buy'].mean()
    utilization_ratio = df['Avg_Utilization_Ratio'].mean()
    count_platinum = df['Card_Category'][df['Card_Category'] == 'platinum'].count()
    count_card = df['Card_Category'].count()
    percent_card = round(count_platinum / count_card, 2)
#print(percent_married)
#create 3 columns
    kpi1, kpi2 = st.columns(2)

#fill in those three columns with respective metrics or KPIs
    kpi1.metric(
        label = 'Average Age',
        value = round(avg_age)
    )

    kpi2.metric(
        label = 'Average Num of Dependents',
        value = round(avg_dep)
    )

    kpi3, kpi4 = st.columns(2)

    # fill in those three columns with respective metrics or KPIs
    kpi4.metric(
        label='Num of Transactions (12-Mon)',
        value=round(avg_trans)
    )

    kpi3.metric(
        label = 'Married Percentage',
        value=percent_married
    )

    kpi5, kpi6 = st.columns(2)

    # fill in those three columns with respective metrics or KPIs
    kpi5.metric(
        label='Transactions Amount (12-Mon)',
        value=round(avg_Amt, 1)
    )

    kpi6.metric(
        label = 'Length of Relationship',
        value=round(avg_relation, 1)
    )

    kpi7, kpi8 = st.columns(2)

    # fill in those three columns with respective metrics or KPIs
    kpi5.metric(
        label='Num of Products Holding',
        value=round(num_prod)
    )

    kpi6.metric(
        label = 'Num of Months being Inactive',
        value=round(mon_inactive)
    )

    kpi9, kpi10 = st.columns(2)

    # fill in those three columns with respective metrics or KPIs
    kpi9.metric(
        label='Num of Products Holding',
        value=round(num_prod)
    )

    kpi10.metric(
        label = 'Num of Months being Inactive',
        value=round(mon_inactive, 1)
    )

    kpi11, kpi12 = st.columns(2)
    # fill in those three columns with respective metrics or KPIs
    kpi11.metric(
        label='Credit Card Credit Limit',
        value=round(credit_lim, 1)
    )

    kpi12.metric(
        label = 'Revolving Balance',
        value=round(revolving, 1)
    )

    kpi13, kpi14 = st.columns(2)
    # fill in those three columns with respective metrics or KPIs
    kpi13.metric(
        label='Open to Buy Credit Line',
        value=round(open_to_buy,1)
    )

    kpi14.metric(
        label = 'Card Utilization Ratio',
        value=round(utilization_ratio,2)
    )
# dataframe filter
df2 = get_data()
edu = df2.groupby('Attrition_Flag', as_index=False)['Education_Level'].value_counts(normalize=True).reset_index()

st.markdown("Histogram: Distribution of Customer Age")
fig = px.histogram(df2, x='Customer_Age')
st.write(fig)

#create two columns for charts

gender_df = df2.groupby('Attrition_Flag', as_index=False)['Gender'].value_counts().reset_index()
st.markdown("Bar chart: Number of customers by gender and customer status")
fig = px.bar(gender_df, x='Gender', y='count', color='Attrition_Flag', barmode='group')
st.write(fig)


st.markdown("Bar chart: Proportion of customers by education level and Customer Status")
fig = px.bar(edu, x='Education_Level', y='proportion', color='Attrition_Flag', barmode='group')
st.write(fig)

marrige = df2.groupby('Attrition_Flag', as_index=False)['Marital_Status'].value_counts(normalize=True).reset_index()
st.markdown("Bar chart: Proportion of customers by Marital Status and Customer Status")
fig = px.bar(marrige, x='Marital_Status', y='proportion', color='Attrition_Flag', barmode='group')
st.write(fig)

income_df = df2.groupby('Attrition_Flag', as_index=False)['Income_Category'].value_counts().reset_index()
st.markdown("Bar chart: Income Category by Customer Status")
fig = px.bar(income_df, x='Income_Category', y='count', color='Attrition_Flag', barmode='group')
st.write(fig)

card = df2.groupby('Attrition_Flag', as_index=False)['Card_Category'].value_counts(normalize=True).reset_index()
st.markdown("Bar chart: Proportion of customers by Card Category and Customer Status")
fig = px.bar(card, x='Card_Category', y='proportion', color='Attrition_Flag', barmode='group')
st.write(fig)



#calculate the aveage credit limit by attrition flag
credit = df2.groupby('Attrition_Flag')['Credit_Limit'].mean().reset_index()
status = credit['Attrition_Flag']
value = credit['Credit_Limit']
# create two columns for charts
fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.markdown("Bar chart of Average Credit Limit by Customer Status")
    fig1 = px.bar(x=status, y = value, text = round(value,2), width=300, height=400)
    st.write(fig1)

trans_df = df2.groupby('Attrition_Flag')['Total_Trans_Amt'].mean().reset_index()
status = trans_df['Attrition_Flag']
value = trans_df['Total_Trans_Amt']

with fig_col2:
    st.markdown("Bar chart of Average Transaction Amount by Customer Status")
    fig2= px.bar(x=status, y = value, text = round(value,2), width=300, height=400)
    st.write(fig2)

trans_ct = df2.groupby('Attrition_Flag')['Total_Trans_Ct'].mean().reset_index()
status = trans_ct['Attrition_Flag']
value = trans_ct['Total_Trans_Ct']
# create two columns for charts
fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.markdown("Bar chart of Average Number of Transactions by Customer Status")
    fig1 = px.bar(x=status, y = value, text = round(value,2), width=300, height=400)
    st.write(fig1)

change = df2.groupby('Attrition_Flag')['Total_Ct_Chng_Q4_Q1'].mean().reset_index()
status = change['Attrition_Flag']
value = change['Total_Ct_Chng_Q4_Q1']

with fig_col2:
    st.markdown("Bar chart of Average Difference in Transaction Amount betwwen Q1 & Q4")
    fig2= px.bar(x=status, y = value, text = round(value,2), width=300, height=400)
    st.write(fig2)

inactive = df2.groupby('Attrition_Flag')['Months_Inactive_12_mon'].mean().reset_index()
status = inactive['Attrition_Flag']
value = inactive['Months_Inactive_12_mon']
# create two columns for charts
fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.markdown("Bar chart of Average number of months the customer is inactive by customer status")
    fig1 = px.bar(x=status, y = value, text = round(value,2), width=300, height=400)
    st.write(fig1)

num_pro = df2.groupby('Attrition_Flag')['Total_Relationship_Count'].mean().reset_index()
status = num_pro['Attrition_Flag']
value = num_pro['Total_Relationship_Count']

with fig_col2:
    st.markdown("Bar chart of Average number of products the customer is holding by Customer status")
    fig2= px.bar(x=status, y = value, text = round(value,2), width=300, height=400)
    st.write(fig2)

length = df2.groupby('Attrition_Flag')['Months_on_book'].mean().reset_index()
status = length['Attrition_Flag']
value = length['Months_on_book']
# create two columns for charts
fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.markdown("Bar chart of Average Length of the Relationship with Bank")
    fig1 = px.bar(x=status, y = value, text = round(value,2), width=300, height=400)
    st.write(fig1)

with fig_col2:
    st.markdown("Boxplot of Difference in Transaction Amount betwwen Q1 & Q4")
    fig2= px.box(df2, x='Attrition_Flag', y = 'Total_Ct_Chng_Q4_Q1', width=300, height=400)
    st.write(fig2)

# create two columns for charts
fig_col1, fig_col2 = st.columns(2)

with fig_col1:
    st.markdown("Box Plot of card utilization ratio")
    fig1= px.box(df2, x='Attrition_Flag', y = 'Avg_Utilization_Ratio', width=300, height=400)
    st.write(fig1)

with fig_col2:
    st.markdown("Box Plot of Total Credit Card Revolving Balance")
    fig2= px.box(df2, x='Attrition_Flag', y = 'Total_Revolving_Bal', width=300, height=400)
    st.write(fig2)

st.markdown("Scatter Plot: Total_Revolving_Bal and Total_Trans_Amt Colored by Customer Status")
fig = px.scatter(df2, x='Total_Revolving_Bal', y='Total_Trans_Amt', color='Attrition_Flag')
st.write(fig)

st.markdown("Scatter Plot: Total_Ct_Chng & Total_Relationship_Count Colored by Customer Status")
fig = px.scatter(df2, x='Total_Ct_Chng_Q4_Q1', y='Total_Relationship_Count', color='Attrition_Flag')
st.write(fig)

st.markdown("Scatter Plot: Dependent_count & Months_on_book Colored by Customer Status")
fig = px.scatter(df2, x='Dependent_count', y='Months_on_book', color='Attrition_Flag')
st.write(fig)

df_cleaned = df.drop(['CLIENTNUM','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1', 'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'], axis=1)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
newdf = df_cleaned.select_dtypes(include=numerics)
st.markdown("Scatter Plot: Dependent_count & Months_on_book Colored by Customer Status")
fig = px.imshow(round(newdf.corr(),1), text_auto=True, width=700, height=700)
st.write(fig)

#More histograms
num_fields = ["Months_on_book", "Total_Trans_Ct", "Total_Trans_Amt"]
title_font = {"family": "arial", "color": "k", "weight": "bold", "size": 14}
axes_font = {"family": "arial", "color": "darkgreen", "weight": "bold", "size": 12}
st.markdown("Histograms")
figure, axes = plt.subplots(1, 3, figsize=(28, 7))

for field in num_fields:
    j = num_fields.index(field)

    sns.distplot(df[field], kde=True, bins=20, color="#13787d", hist_kws=dict(linewidth=2, edgecolor="white"),
                 ax=axes[j])

    axes[j].axvline(df[field].mean(), color="r", alpha=0.5, ls="--", lw=2)
    axes[j].axvline(df[field].median(), color="b", alpha=0.5, ls="--", lw=2)

    axes[j].text(x=df[field].mean(), y=(axes[j].get_ylim()[1]) / 2,
                 s=f" Mean = {round(df[field].mean(), 2)}", color="red", weight="bold")
    axes[j].text(x=df[field].median(), y=(axes[j].get_ylim()[1]) / 2.2,
                 s=f" Median = {round(df[field].median(), 2)}", color="blue", weight="bold")

    axes[j].set_title(field + "\n", fontdict=title_font)
    axes[j].set_xlabel(field, fontdict=axes_font)
    axes[j].set_ylabel("Density", fontdict=axes_font)

    # axes[j].annotate("Shapiro-Wilk Test Results\n-----------------------------------",
    # xy = (0.6, 0.8), xycoords = "axes fraction", fontsize = 10,
    # horizontalalignment = "left", verticalalignment = "bottom", weight = "bold")
    # axes[j].annotate(f"Statistics = {round(shapiro(df[field])[0], 4)}\nP-Value = {round(shapiro(df[field])[1], 4)}",
    # xy = (0.6, 0.75), xycoords = "axes fraction", fontsize = 10,
    # horizontalalignment = "left", verticalalignment = "bottom")

num_fields = ["Months_on_book", "Total_Trans_Ct", "Total_Trans_Amt"]
title_font = {"family": "arial", "color": "k", "weight": "bold", "size": 14}
axes_font = {"family": "arial", "color": "darkgreen", "weight": "bold", "size": 12}



figure, axes = plt.subplots(3, 1, figsize=(8, 20))

for field in num_fields:
    j = num_fields.index(field)

    sns.distplot(df[field], kde=True, bins=20, color="#13787d", hist_kws=dict(linewidth=2, edgecolor="white"),
                 ax=axes[j])

    axes[j].axvline(df[field].mean(), color="r", alpha=0.5, ls="--", lw=2)
    axes[j].axvline(df[field].median(), color="b", alpha=0.5, ls="--", lw=2)

    axes[j].text(x=df[field].mean(), y=(axes[j].get_ylim()[1]) / 2,
                 s=f" Mean = {round(df[field].mean(), 2)}", color="red", weight="bold")
    axes[j].text(x=df[field].median(), y=(axes[j].get_ylim()[1]) / 2.2,
                 s=f" Median = {round(df[field].median(), 2)}", color="blue", weight="bold")

    axes[j].set_title(field + "\n", fontdict=title_font)
    axes[j].set_xlabel(field, fontdict=axes_font)
    axes[j].set_ylabel("Density", fontdict=axes_font)

    # axes[j].annotate("Shapiro-Wilk Test Results\n-----------------------------------",
    # xy = (0.6, 0.8), xycoords = "axes fraction", fontsize = 10,
    # horizontalalignment = "left", verticalalignment = "bottom", weight = "bold")
    # axes[j].annotate(f"Statistics = {round(shapiro(df[field])[0], 4)}\nP-Value = {round(shapiro(df[field])[1], 4)}",
    # xy = (0.6, 0.75), xycoords = "axes fraction", fontsize = 10,
    # horizontalalignment = "left", verticalalignment = "bottom")

st.write(figure)
##

# Pie Charts
categ = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category']

colors = ["gold", "mediumturquoise", "navajowhite", "darkorange",
          "lightgreen", "lightseagreen", "lightcoral", "lightslategrey"]

df["Attrition_Flag"].replace([0, 1], ["No", "Yes"], inplace=True)
#st.markdown("Pie Charts: Demographic Info")
fig = make_subplots(rows=1, cols=4, specs=[[{"type": "pie"}] * 4],
                    shared_yaxes=True, subplot_titles=categ[:4])

for field in categ[0:4]:
    i = categ.index(field) + 1
    colors_ = colors[2 * i - 2:  2 * i]

    data = list(df[field])
    labels = df[field].unique()
    values = [data.count(labels[i]) for i in range(len(labels))]

    fig.add_trace(go.Pie(
        labels=labels,
        values=values,
        domain=dict(x=[0, 0.5]),
        name=field,
        marker=dict(colors=colors_, line=dict(color="white", width=2)),
        textinfo="label+percent",
        hovertemplate="<b>Quantity: </b> %{value} <br>",
        textfont_size=12,
        showlegend=False),
        row=1, col=i)

    fig.update_layout(title=dict(text="<b>Customers Demographic Info\b", x=0.01, y=0.96,
                                 font={"family": "arial", "size": 28, "color": "#252525"}),
                      title_font_size=30)

    st.write(fig)
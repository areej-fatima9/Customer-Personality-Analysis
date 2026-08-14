# ============================================================
# MODULE: CUSTOMER PERSONALITY ANALYSIS
# DAY 2 – DASHBOARD DEVELOPMENT
# ============================================================

import os
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Customer Personality Analysis",
    page_icon="👥",
    layout="wide"
)


# ============================================================
# 2. PROJECT PATH
# ============================================================

PROJECT_ROOT = r"D:\aiLAB99\Customer-Personality-Analysis"


# ============================================================
# 3. FIND CSV FILES AUTOMATICALLY
# ============================================================

@st.cache_data
def find_csv_files():

    csv_files = []

    for root, dirs, files in os.walk(PROJECT_ROOT):

        dirs[:] = [
            d for d in dirs
            if d not in ["venv", ".git", "__pycache__"]
        ]

        for file in files:

            if file.lower().endswith(".csv"):

                csv_files.append(
                    os.path.join(root, file)
                )

    return csv_files


csv_files = find_csv_files()


# ============================================================
# 4. FIND CUSTOMER PROFILING DATASET
# ============================================================

@st.cache_data
def find_profiling_dataset():

    # First, look specifically for the customer-level
    # profiling dataset created in Module 7.

    preferred_files = [
        "customer_cluster_profile_summary.csv",
        "cluster_business_profile.csv",
        "baseline_kmeans_clustered_data.csv"
    ]

    # Check preferred files first
    for preferred_name in preferred_files:

        for path in csv_files:

            if os.path.basename(path).lower() == preferred_name.lower():

                try:

                    sample = pd.read_csv(
                        path,
                        nrows=5
                    )

                    if "KMeans_Cluster" in sample.columns:

                        return path

                except Exception:

                    pass


    # If preferred file is not found,
    # search for a large customer-level CSV.

    candidates = []

    for path in csv_files:

        try:

            sample = pd.read_csv(
                path,
                nrows=5
            )

            if "KMeans_Cluster" not in sample.columns:
                continue

            # Get number of rows
            row_count = sum(
                1
                for _ in open(
                    path,
                    "r",
                    encoding="utf-8",
                    errors="ignore"
                )
            ) - 1

            # Customer-level data should contain
            # many customers, not just 2 cluster rows.

            if row_count > 500:

                candidates.append(
                    (path, row_count)
                )

        except Exception:

            continue


    if candidates:

        # Select the largest suitable customer dataset
        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return candidates[0][0]


    return None

# ============================================================
# 5. LOAD MODULE 7 CUSTOMER PROFILING DATASET
# ============================================================

profiling_path = os.path.join(
    PROJECT_ROOT,
    "Cluster_Evaluation_Customer_Profiling",
    "Reports",
    "module_07_customer_cluster_profiling_dataset.csv"
)

# Check file
if not os.path.exists(profiling_path):

    st.error(
        "Module 7 customer profiling dataset was not found."
    )

    st.code(profiling_path)

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data(path):
    return pd.read_csv(path)


# IMPORTANT: Load dataframe BEFORE using df
df = load_data(profiling_path)


# ============================================================
# VERIFY DATA
# ============================================================

if df.empty:

    st.error("The Module 7 dataset is empty.")
    st.stop()


if "KMeans_Cluster" not in df.columns:

    st.error(
        "KMeans_Cluster column was not found."
    )

    st.write(
        "Available columns:",
        df.columns.tolist()
    )

    st.stop()


# ============================================================
# DATASET STATUS
# ============================================================

st.sidebar.success(
    f"Customer dataset loaded: {len(df):,} customers"
)


# ============================================================
# BASIC DATA PREPARATION
# ============================================================

df["KMeans_Cluster"] = df["KMeans_Cluster"].astype(int)


# ============================================================
# BASIC INFORMATION
# ============================================================

total_customers = len(df)

total_segments = df["KMeans_Cluster"].nunique()


st.sidebar.write(
    f"**Customers:** {total_customers:,}"
)

st.sidebar.write(
    f"**Segments:** {total_segments}"
)


# ============================================================
# CLUSTER DISTRIBUTION
# ============================================================

cluster_counts = (
    df["KMeans_Cluster"]
    .value_counts()
    .sort_index()
)

cluster_percentages = (
    df["KMeans_Cluster"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)
# ============================================================
# 6. BASIC DATA PREPARATION
# ============================================================

df.columns = [
    str(col).strip()
    for col in df.columns
]


# Convert cluster to string for filtering

df["KMeans_Cluster"] = (
    df["KMeans_Cluster"]
    .astype(str)
)


# ============================================================
# 7. BUSINESS SEGMENT NAMES
# ============================================================

segment_names = {

    "0": "Budget-Conscious Family Buyers",

    "1": "High-Value Premium Customers"
}


df["Segment_Name"] = (
    df["KMeans_Cluster"]
    .map(segment_names)
    .fillna(
        "Customer Segment "
        + df["KMeans_Cluster"]
    )
)


# ============================================================
# 8. SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("📊 Dashboard Navigation")

page = st.sidebar.radio(

    "Select Dashboard Section",

    [
        "🏠 Overview",
        "👥 Customer Segments",
        "💰 Spending Analysis",
        "🛒 Shopping Behavior",
        "📢 Campaign Analysis",
        "💡 Business Recommendations",
        "📋 Customer Data"
    ]
)


# ============================================================
# 9. SIDEBAR FILTERS
# ============================================================

st.sidebar.markdown("---")

st.sidebar.subheader("🔎 Filters")


# Segment selection

segment_options = [
    "All"
] + sorted(
    df["Segment_Name"]
    .dropna()
    .unique()
    .tolist()
)


selected_segment = st.sidebar.selectbox(

    "Select Segment",

    segment_options
)


# Customer search

customer_search = st.sidebar.text_input(

    "Search Customer ID",

    placeholder="Enter Customer ID"
)


# ============================================================
# 10. APPLY FILTERS
# ============================================================

filtered_df = df.copy()


if selected_segment != "All":

    filtered_df = filtered_df[
        filtered_df["Segment_Name"]
        == selected_segment
    ]


if customer_search:

    if "ID" in filtered_df.columns:

        filtered_df = filtered_df[
            filtered_df["ID"]
            .astype(str)
            .str.contains(
                customer_search,
                case=False,
                na=False
            )
        ]


# ============================================================
# 11. COMMON KPI VALUES
# ============================================================

customer_count = len(filtered_df)


if "Income" in filtered_df.columns:

    avg_income = filtered_df["Income"].mean()

else:

    avg_income = 0


if "Total_Spending" in filtered_df.columns:

    avg_spending = (
        filtered_df["Total_Spending"]
        .mean()
    )

else:

    avg_spending = 0


if "Total_Purchases" in filtered_df.columns:

    avg_purchases = (
        filtered_df["Total_Purchases"]
        .mean()
    )

else:

    avg_purchases = 0


# ============================================================
# 12. HEADER
# ============================================================

st.title(
    "👥 Customer Personality Analysis Dashboard"
)

st.caption(
    "Machine Learning Customer Segmentation "
    "and Business Insights"
)


# ============================================================
# PAGE 1 – OVERVIEW
# ============================================================

if page == "🏠 Overview":

    st.header(
        "Customer Segmentation Overview"
    )

    # KPI cards

    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Customers",
        f"{customer_count:,}"
    )


    col2.metric(
        "Segments",
        df["KMeans_Cluster"].nunique()
    )


    col3.metric(
        "Average Income",
        f"{avg_income:,.2f}"
    )


    col4.metric(
        "Average Spending",
        f"{avg_spending:,.2f}"
    )


    st.markdown("---")


    # Segment distribution

    st.subheader(
        "Customer Segment Distribution"
    )


    segment_counts = (

        filtered_df[
            "Segment_Name"
        ]

        .value_counts()

        .reset_index()
    )


    segment_counts.columns = [
        "Segment",
        "Customers"
    ]


    fig = px.bar(

        segment_counts,

        x="Segment",

        y="Customers",

        title="Customers by Segment",

        text="Customers"
    )


    fig.update_layout(
        xaxis_title="Customer Segment",
        yaxis_title="Number of Customers"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Segment table

    st.subheader(
        "Segment Summary"
    )


    summary = (

        filtered_df

        .groupby(
            "Segment_Name"
        )

        .size()

        .reset_index(
            name="Customers"
        )
    )


    summary["Percentage"] = (

        summary["Customers"]

        / len(filtered_df)

        * 100

    ).round(2)


    st.dataframe(
        summary,
        use_container_width=True
    )


# ============================================================
# PAGE 2 – CUSTOMER SEGMENTS
# ============================================================

elif page == "👥 Customer Segments":

    st.header(
        "Customer Segments"
    )


    segment_summary = (

        filtered_df

        .groupby(
            "Segment_Name"
        )

        .agg(
            Customers=(
                "KMeans_Cluster",
                "count"
            )
        )

        .reset_index()
    )


    segment_summary["Percentage"] = (

        segment_summary["Customers"]

        / len(filtered_df)

        * 100

    ).round(2)


    st.dataframe(
        segment_summary,
        use_container_width=True
    )


    st.subheader(
        "Cluster Distribution"
    )


    fig = px.pie(

        segment_summary,

        names="Segment_Name",

        values="Customers",

        title="Customer Distribution by Segment"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    # Income comparison

    if "Income" in filtered_df.columns:

        st.subheader(
            "Income Comparison"
        )


        income_df = (

            filtered_df

            .groupby(
                "Segment_Name"
            )["Income"]

            .mean()

            .reset_index()
        )


        fig = px.bar(

            income_df,

            x="Segment_Name",

            y="Income",

            text_auto=".2f",

            title="Average Income by Segment"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# PAGE 3 – SPENDING ANALYSIS
# ============================================================

elif page == "💰 Spending Analysis":

    st.header(
        "Spending Analysis"
    )


    if "Total_Spending" in filtered_df.columns:

        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "Average Total Spending",
                f"{filtered_df['Total_Spending'].mean():,.2f}"
            )


        with col2:

            st.metric(
                "Maximum Total Spending",
                f"{filtered_df['Total_Spending'].max():,.2f}"
            )


        spending_df = (

            filtered_df

            .groupby(
                "Segment_Name"
            )["Total_Spending"]

            .mean()

            .reset_index()
        )


        fig = px.bar(

            spending_df,

            x="Segment_Name",

            y="Total_Spending",

            text_auto=".2f",

            title="Average Spending by Segment"
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # Product spending

    product_features = [

        "MntWines",

        "MntFruits",

        "MntMeatProducts",

        "MntFishProducts",

        "MntSweetProducts",

        "MntGoldProds"
    ]


    available_products = [

        col

        for col in product_features

        if col in filtered_df.columns
    ]


    if available_products:

        st.subheader(
            "Product Category Spending"
        )


        product_summary = (

            filtered_df[
                available_products
            ]

            .mean()

            .reset_index()
        )


        product_summary.columns = [
            "Product",
            "Average_Spending"
        ]


        fig = px.bar(

            product_summary,

            x="Product",

            y="Average_Spending",

            title="Average Spending by Product Category"
        )


        fig.update_layout(
            xaxis_tickangle=-45
        )


        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# PAGE 4 – SHOPPING BEHAVIOR
# ============================================================

elif page == "🛒 Shopping Behavior":

    st.header(
        "Shopping Behavior Analysis"
    )


    channel_features = [

        "NumWebPurchases",

        "NumCatalogPurchases",

        "NumStorePurchases",

        "NumDealsPurchases"
    ]


    available_channels = [

        col

        for col in channel_features

        if col in filtered_df.columns
    ]


    if available_channels:

        channel_summary = (

            filtered_df[
                available_channels
            ]

            .mean()

            .reset_index()
        )


        channel_summary.columns = [
            "Channel",
            "Average_Purchases"
        ]


        fig = px.bar(

            channel_summary,

            x="Channel",

            y="Average_Purchases",

            text_auto=".2f",

            title="Average Purchases by Channel"
        )


        fig.update_layout(
            xaxis_tickangle=-30
        )


        st.plotly_chart(

            fig,

            use_container_width=True
        )


    # Website visits

    if "NumWebVisitsMonth" in filtered_df.columns:

        st.subheader(
            "Website Engagement"
        )


        st.metric(

            "Average Website Visits",

            f"{filtered_df['NumWebVisitsMonth'].mean():.2f}"
        )


    # Recency

    if "Recency" in filtered_df.columns:

        st.subheader(
            "Customer Recency"
        )


        st.metric(

            "Average Recency",

            f"{filtered_df['Recency'].mean():.2f} days"
        )


# ============================================================
# PAGE 5 – CAMPAIGN ANALYSIS
# ============================================================

elif page == "📢 Campaign Analysis":

    st.header(
        "Marketing Campaign Analysis"
    )


    campaign_features = [

        "AcceptedCmp1",

        "AcceptedCmp2",

        "AcceptedCmp3",

        "AcceptedCmp4",

        "AcceptedCmp5",

        "Response"
    ]


    available_campaigns = [

        col

        for col in campaign_features

        if col in filtered_df.columns
    ]


    if available_campaigns:

        campaign_summary = (

            filtered_df[
                available_campaigns
            ]

            .mean()

            .mul(100)

            .reset_index()
        )


        campaign_summary.columns = [

            "Campaign",

            "Response_Rate"
        ]


        fig = px.bar(

            campaign_summary,

            x="Campaign",

            y="Response_Rate",

            text_auto=".2f",

            title="Campaign Response Rate (%)"
        )


        fig.update_layout(

            yaxis_title="Response Rate (%)",

            xaxis_title="Campaign"
        )


        st.plotly_chart(

            fig,

            use_container_width=True
        )


        st.dataframe(

            campaign_summary,

            use_container_width=True
        )


    # Complaints

    if "Complain" in filtered_df.columns:

        complaint_rate = (

            filtered_df["Complain"]

            .mean()

            * 100
        )


        st.metric(

            "Complaint Rate",

            f"{complaint_rate:.2f}%"
        )


# ============================================================
# PAGE 6 – BUSINESS RECOMMENDATIONS
# ============================================================

elif page == "💡 Business Recommendations":

    st.header(
        "Business Recommendations"
    )


    if selected_segment == "All":

        st.info(
            "Select a specific customer segment "
            "from the sidebar to view targeted recommendations."
        )


    else:

        st.subheader(
            selected_segment
        )


        if selected_segment == (
            "High-Value Premium Customers"
        ):

            st.success(
                """
                **Recommended Strategy**

                • Focus on customer loyalty and retention.

                • Provide premium and personalized offers.

                • Encourage upselling and cross-selling.

                • Recommend premium product categories.

                • Use personalized email campaigns.

                • Reward repeat purchases.
                """
            )


        elif selected_segment == (
            "Budget-Conscious Family Buyers"
        ):

            st.info(
                """
                **Recommended Strategy**

                • Use value-based promotional offers.

                • Provide bundle discounts.

                • Promote family-oriented products.

                • Use suitable coupons and deals.

                • Encourage repeat purchases through promotions.

                • Use reactivation campaigns when engagement declines.
                """
            )


        else:

            st.write(
                "Use customer behavior, spending, "
                "engagement and campaign response "
                "to develop a personalized strategy."
            )


# ============================================================
# PAGE 7 – CUSTOMER DATA
# ============================================================

elif page == "📋 Customer Data":

    st.header(
        "Customer Data"
    )


    st.write(
        f"Showing {len(filtered_df):,} customers."
    )


    st.dataframe(

        filtered_df,

        use_container_width=True,

        height=500
    )


    # Download

    csv_download = filtered_df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )


    st.download_button(

        label="⬇️ Download Filtered Customer Data",

        data=csv_download,

        file_name="filtered_customer_data.csv",

        mime="text/csv"
    )


# ============================================================
# 13. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Customer Personality Analysis | "
    "Machine Learning Customer Segmentation Dashboard"
)

# ============================================================
# DAY 3 — MODEL / APPLICATION INTEGRATION
# ============================================================

import joblib
import numpy as np


# ============================================================
# 1. MODEL FILE PATHS
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "Customer_Segmentation",
    "Models",
    "final_kmeans_model.pkl"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "Customer_Segmentation",
    "Models",
    "standard_scaler.pkl"
)


# ============================================================
# 2. LOAD SAVED MODEL AND SCALER
# ============================================================

@st.cache_resource
def load_ml_objects():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


try:

    final_model, final_scaler = load_ml_objects()

except Exception as e:

    st.error(f"Could not load saved ML files: {e}")
    st.stop()


# ============================================================
# 3. VERIFY MODEL
# ============================================================

model_features = getattr(
    final_model,
    "n_features_in_",
    None
)

scaler_features = getattr(
    final_scaler,
    "n_features_in_",
    None
)


if model_features != 17:

    st.error(
        f"Final K-Means model expects "
        f"{model_features} features instead of 17."
    )

    st.stop()


if scaler_features != 17:

    st.error(
        f"StandardScaler expects "
        f"{scaler_features} features instead of 17."
    )

    st.stop()


# ============================================================
# 4. DAY 3 SECTION
# ============================================================

st.header("🤖 Customer Segment Prediction")

st.write(
    "Enter customer information and use the saved "
    "K-Means model to predict the customer segment."
)


# ============================================================
# 5. CUSTOMER INPUT
# ============================================================

col1, col2 = st.columns(2)


with col1:

    customer_age = st.number_input(
        "Customer Age",
        min_value=18,
        max_value=100,
        value=40
    )

    income = st.number_input(
        "Income",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

    total_children = st.number_input(
        "Total Children",
        min_value=0,
        max_value=10,
        value=1
    )

    family_size = st.number_input(
        "Family Size",
        min_value=1,
        max_value=10,
        value=2
    )

    total_spending = st.number_input(
        "Total Spending",
        min_value=0.0,
        value=500.0,
        step=50.0
    )

    total_purchases = st.number_input(
        "Total Purchases",
        min_value=0,
        value=10
    )

    web_purchases = st.number_input(
        "Web Purchases",
        min_value=0,
        value=4
    )

    catalog_purchases = st.number_input(
        "Catalog Purchases",
        min_value=0,
        value=2
    )

    store_purchases = st.number_input(
        "Store Purchases",
        min_value=0,
        value=5
    )


with col2:

    campaign_acceptance = st.number_input(
        "Total Campaign Acceptance",
        min_value=0,
        max_value=5,
        value=0
    )

    digital_engagement = st.number_input(
        "Digital Engagement",
        min_value=0.0,
        value=5.0
    )

    web_visits = st.number_input(
        "Web Visits per Month",
        min_value=0,
        max_value=30,
        value=5
    )

    recency = st.number_input(
        "Recency",
        min_value=0,
        max_value=100,
        value=30
    )

    customer_tenure = st.number_input(
        "Customer Tenure (Years)",
        min_value=0.0,
        value=2.0
    )

    deal_dependency = st.number_input(
        "Deal Dependency",
        min_value=0.0,
        value=0.2
    )

    deals_purchases = st.number_input(
        "Deal Purchases",
        min_value=0,
        value=2
    )


# ============================================================
# 6. ENGINEERED FEATURE
# ============================================================

if total_purchases > 0:

    average_spending = (
        total_spending / total_purchases
    )

else:

    average_spending = 0.0


# ============================================================
# 7. CREATE 17 FEATURES
# ============================================================

input_data = pd.DataFrame({

    "numerical__Customer_Age": [
        customer_age
    ],

    "numerical__Income": [
        income
    ],

    "numerical__Total_Children": [
        total_children
    ],

    "numerical__Family_Size": [
        family_size
    ],

    "numerical__Total_Spending": [
        total_spending
    ],

    "numerical__Average_Spending_per_Purchase": [
        average_spending
    ],

    "numerical__Total_Purchases": [
        total_purchases
    ],

    "numerical__NumWebPurchases": [
        web_purchases
    ],

    "numerical__NumCatalogPurchases": [
        catalog_purchases
    ],

    "numerical__NumStorePurchases": [
        store_purchases
    ],

    "numerical__Total_Campaign_Acceptance": [
        campaign_acceptance
    ],

    "numerical__Digital_Engagement": [
        digital_engagement
    ],

    "numerical__NumWebVisitsMonth": [
        web_visits
    ],

    "numerical__Recency": [
        recency
    ],

    "numerical__Customer_Tenure_Years": [
        customer_tenure
    ],

    "numerical__Deal_Dependency": [
        deal_dependency
    ],

    "numerical__NumDealsPurchases": [
        deals_purchases
    ]
})


# ============================================================
# 8. EXACT FEATURE ORDER
# ============================================================

required_features = [

    "numerical__Customer_Age",
    "numerical__Income",
    "numerical__Total_Children",
    "numerical__Family_Size",
    "numerical__Total_Spending",
    "numerical__Average_Spending_per_Purchase",
    "numerical__Total_Purchases",
    "numerical__NumWebPurchases",
    "numerical__NumCatalogPurchases",
    "numerical__NumStorePurchases",
    "numerical__Total_Campaign_Acceptance",
    "numerical__Digital_Engagement",
    "numerical__NumWebVisitsMonth",
    "numerical__Recency",
    "numerical__Customer_Tenure_Years",
    "numerical__Deal_Dependency",
    "numerical__NumDealsPurchases"

]


input_data = input_data[required_features]


# ============================================================
# 9. PREDICT SEGMENT
# ============================================================

if st.button(
    "🔍 Predict Customer Segment",
    type="primary"
):

    try:

        scaled_input = final_scaler.transform(
            input_data
        )

        predicted_cluster = int(
            final_model.predict(
                scaled_input
            )[0]
        )

        # ====================================================
        # 10. SHOW RESULT
        # ====================================================

        st.success(
            f"Customer belongs to Cluster {predicted_cluster}"
        )

        # ====================================================
        # 11. SEGMENT CHARACTERISTICS
        # ====================================================

        if "KMeans_Cluster" in df.columns:

            # Convert cluster values to numeric for reliable matching
            cluster_values = pd.to_numeric(
                df["KMeans_Cluster"],
                errors="coerce"
            )

            # Convert predicted cluster to integer
            predicted_cluster_int = int(predicted_cluster)

            # Select customers belonging to predicted cluster
            cluster_data = df[
                cluster_values == predicted_cluster_int
            ].copy()

            st.subheader(
                "📊 Segment Characteristics"
            )

            if len(cluster_data) > 0:

                c1, c2, c3 = st.columns(3)

                # ------------------------------------------------
                # Customer Count
                # ------------------------------------------------

                with c1:

                    st.metric(
                        "Customers",
                        f"{len(cluster_data):,}"
                    )

                # ------------------------------------------------
                # Average Income
                # ------------------------------------------------

                with c2:

                    if "Income" in cluster_data.columns:

                        income_values = pd.to_numeric(
                            cluster_data["Income"],
                            errors="coerce"
                        )

                        avg_income = income_values.mean()

                        st.metric(
                            "Average Income",
                            f"${avg_income:,.0f}"
                            if pd.notna(avg_income)
                            else "$0"
                        )

                    else:

                        st.metric(
                            "Average Income",
                            "N/A"
                        )

                # ------------------------------------------------
                # Average Age
                # ------------------------------------------------

                with c3:

                    # Prefer Customer_Age if available
                    if "Customer_Age" in cluster_data.columns:

                        age_values = pd.to_numeric(
                            cluster_data["Customer_Age"],
                            errors="coerce"
                        )

                    elif "Age" in cluster_data.columns:

                        age_values = pd.to_numeric(
                            cluster_data["Age"],
                            errors="coerce"
                        )

                    else:

                        age_values = pd.Series(dtype=float)

                    avg_age = age_values.mean()

                    st.metric(
                        "Average Age",
                        f"{avg_age:.1f}"
                        if pd.notna(avg_age)
                        else "N/A"
                    )

            else:

                st.warning(
                    f"No customers found for Cluster "
                    f"{predicted_cluster_int}."
                )

        else:

            st.warning(
                "KMeans_Cluster column is not available."
            )

        # ====================================================
        # 12. MARKETING RECOMMENDATION
        # ====================================================

        st.subheader(
            "💡 Marketing Recommendation"
        )

        if predicted_cluster == 0:

            st.info(
                """
                **Cluster 0 — Family-Oriented Customers**

                • Promote family-friendly products  
                • Provide value-based offers  
                • Encourage repeat purchases  
                • Recommend product bundles  
                • Use personalized campaigns
                """
            )

        elif predicted_cluster == 1:

            st.info(
                """
                **Cluster 1 — High-Value Customers**

                • Promote premium products  
                • Provide exclusive offers  
                • Encourage loyalty  
                • Use personalized recommendations  
                • Focus on upselling
                """
            )

        # ====================================================
        # 13. SHOW FEATURES
        # ====================================================

        with st.expander(
            "View Features Used for Prediction"
        ):

            st.dataframe(
                input_data,
                use_container_width=True
            )

        # ====================================================
        # 14. PASS MESSAGE
        # ====================================================

        st.success(
            "DAY 3 MODEL INTEGRATION: PASS"
        )

    except Exception as e:

        st.error(
            f"Prediction failed: {e}"
        )
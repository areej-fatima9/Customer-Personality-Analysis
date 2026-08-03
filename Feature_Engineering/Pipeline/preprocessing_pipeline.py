import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib


# ============================================================
# TASK 7
# PREPROCESSING PIPELINE DEVELOPMENT
# Customer Personality Segmentation Project
# ============================================================


# ============================================================
# 1. LOAD DATA
# ============================================================

def load_data(file_path):
    """
    Load the cleaned customer personality dataset.
    """

    print("=" * 60)
    print("STEP 1: LOADING DATA")
    print("=" * 60)

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully.")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])
    print()

    return df


# ============================================================
# 2. REMOVE DUPLICATES
# ============================================================

def remove_duplicates(df):
    """
    Remove duplicate records and duplicate Customer IDs.
    """

    print("=" * 60)
    print("STEP 2: REMOVING DUPLICATES")
    print("=" * 60)

    before_rows = len(df)

    # Remove completely duplicated rows
    df = df.drop_duplicates()

    after_duplicate_rows = len(df)

    print(
        "Duplicate rows removed:",
        before_rows - after_duplicate_rows
    )

    # Remove duplicate Customer IDs
    if "ID" in df.columns:

        before_id_rows = len(df)

        df = df.drop_duplicates(
            subset=["ID"],
            keep="first"
        )

        after_id_rows = len(df)

        print(
            "Duplicate Customer IDs removed:",
            before_id_rows - after_id_rows
        )

    print("Rows after duplicate removal:", len(df))
    print()

    return df


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

def handle_missing_values(df):
    """
    Handle missing numerical values using median imputation.
    """

    print("=" * 60)
    print("STEP 3: HANDLING MISSING VALUES")
    print("=" * 60)

    missing_before = df.isnull().sum().sum()

    print(
        "Missing values before processing:",
        missing_before
    )

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    for column in numerical_columns:

        if df[column].isnull().any():

            df[column] = df[column].fillna(
                df[column].median()
            )

    # Handle missing categorical values
    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        if df[column].isnull().any():

            mode_value = df[column].mode()

            if len(mode_value) > 0:

                df[column] = df[column].fillna(
                    mode_value[0]
                )

    missing_after = df.isnull().sum().sum()

    print(
        "Missing values after processing:",
        missing_after
    )

    print()

    return df


# ============================================================
# 4. CREATE ENGINEERED FEATURES
# ============================================================

def create_features(df):
    """
    Create behavioral and demographic customer features.
    """

    print("=" * 60)
    print("STEP 4: FEATURE CREATION")
    print("=" * 60)

    df = df.copy()

    # --------------------------------------------------------
    # Customer Age
    # --------------------------------------------------------

    current_year = pd.Timestamp.now().year

    df["Customer_Age"] = (
        current_year - df["Year_Birth"]
    )

    # --------------------------------------------------------
    # Customer Tenure
    # --------------------------------------------------------

    df["Dt_Customer"] = pd.to_datetime(
        df["Dt_Customer"],
        errors="coerce"
    )

    latest_date = df["Dt_Customer"].max()

    df["Customer_Tenure_Years"] = (
        (latest_date - df["Dt_Customer"]).dt.days
        / 365.25
    )

    # --------------------------------------------------------
    # Total Children
    # --------------------------------------------------------

    df["Total_Children"] = (
        df["Kidhome"] +
        df["Teenhome"]
    )

    # --------------------------------------------------------
    # Family Size
    # --------------------------------------------------------

    df["Family_Size"] = (
        1 +
        df["Total_Children"]
    )

    # --------------------------------------------------------
    # Total Spending
    # --------------------------------------------------------

    spending_columns = [
        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds"
    ]

    df["Total_Spending"] = df[
        spending_columns
    ].sum(axis=1)

    # --------------------------------------------------------
    # Total Purchases
    # --------------------------------------------------------

    purchase_columns = [
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases"
    ]

    df["Total_Purchases"] = df[
        purchase_columns
    ].sum(axis=1)

    # --------------------------------------------------------
    # Campaign Acceptance
    # --------------------------------------------------------

    campaign_columns = [
        "AcceptedCmp1",
        "AcceptedCmp2",
        "AcceptedCmp3",
        "AcceptedCmp4",
        "AcceptedCmp5"
    ]

    df["Total_Campaign_Acceptance"] = df[
        campaign_columns
    ].sum(axis=1)

    # --------------------------------------------------------
    # Average Spending per Purchase
    # --------------------------------------------------------

    df["Average_Spending_per_Purchase"] = (
        df["Total_Spending"] /
        df["Total_Purchases"].replace(0, np.nan)
    )

    df["Average_Spending_per_Purchase"] = (
        df["Average_Spending_per_Purchase"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Digital Engagement
    # --------------------------------------------------------

    df["Digital_Engagement"] = (
        df["NumWebPurchases"] +
        df["NumWebVisitsMonth"]
    )

    # --------------------------------------------------------
    # Deal Dependency
    # --------------------------------------------------------

    df["Deal_Dependency"] = (
        df["NumDealsPurchases"] /
        df["Total_Purchases"].replace(0, np.nan)
    )

    df["Deal_Dependency"] = (
        df["Deal_Dependency"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Preferred Shopping Channel
    # --------------------------------------------------------

    channel_columns = [
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases"
    ]

    channel_names = {
        "NumWebPurchases": "Web",
        "NumCatalogPurchases": "Catalog",
        "NumStorePurchases": "Store"
    }

    df["Preferred_Shopping_Channel"] = (
        df[channel_columns]
        .idxmax(axis=1)
        .map(channel_names)
    )

    # --------------------------------------------------------
    # Product Preference
    # --------------------------------------------------------

    product_columns = {
        "MntWines": "Wine",
        "MntFruits": "Fruits",
        "MntMeatProducts": "Meat",
        "MntFishProducts": "Fish",
        "MntSweetProducts": "Sweets",
        "MntGoldProds": "Gold"
    }

    product_column_list = list(
        product_columns.keys()
    )

    df["Product_Preference"] = (
        df[product_column_list]
        .idxmax(axis=1)
        .map(product_columns)
    )

    # --------------------------------------------------------
    # Customer Activity Level
    # --------------------------------------------------------

    activity_score = (
        df["Total_Purchases"] +
        df["Digital_Engagement"] +
        df["Total_Campaign_Acceptance"]
    )

    df["Customer_Activity_Level"] = pd.cut(
        activity_score,
        bins=[
            -np.inf,
            activity_score.quantile(0.33),
            activity_score.quantile(0.66),
            np.inf
        ],
        labels=[
            "Inactive",
            "Moderately Active",
            "Highly Active"
        ],
        include_lowest=True
    )

    print("Feature creation completed.")

    print("\nNew engineered features:")

    engineered_features = [
        "Customer_Age",
        "Customer_Tenure_Years",
        "Total_Children",
        "Family_Size",
        "Total_Spending",
        "Total_Purchases",
        "Total_Campaign_Acceptance",
        "Average_Spending_per_Purchase",
        "Digital_Engagement",
        "Deal_Dependency",
        "Preferred_Shopping_Channel",
        "Product_Preference",
        "Customer_Activity_Level"
    ]

    for feature in engineered_features:

        print(" -", feature)

    print()

    return df


# ============================================================
# 5. SELECT FEATURES
# ============================================================

def select_features(df):
    """
    Select features relevant for customer segmentation.
    """

    print("=" * 60)
    print("STEP 5: FEATURE SELECTION")
    print("=" * 60)

    selected_features = [

        # Demographics
        "Customer_Age",
        "Income",

        # Household
        "Total_Children",
        "Family_Size",

        # Spending
        "Total_Spending",
        "Average_Spending_per_Purchase",

        "MntWines",
        "MntFruits",
        "MntMeatProducts",
        "MntFishProducts",
        "MntSweetProducts",
        "MntGoldProds",

        # Purchases
        "Total_Purchases",
        "NumWebPurchases",
        "NumCatalogPurchases",
        "NumStorePurchases",

        # Campaign
        "Total_Campaign_Acceptance",
        "Response",

        # Engagement
        "Digital_Engagement",
        "NumWebVisitsMonth",

        # Activity
        "Recency",
        "Customer_Tenure_Years",
        "Deal_Dependency",
        "NumDealsPurchases",

        # Categorical features
        "Education",
        "Marital_Status",
        "Preferred_Shopping_Channel",
        "Product_Preference",
        "Customer_Activity_Level"
    ]

    # Keep only columns that actually exist
    selected_features = [
        column
        for column in selected_features
        if column in df.columns
    ]

    selected_df = df[
        selected_features
    ].copy()

    print(
        "Number of selected features:",
        len(selected_features)
    )

    print("\nSelected features:")

    for feature in selected_features:
        print(" -", feature)

    print()

    return selected_df


# ============================================================
# 6. LOG TRANSFORMATION
# ============================================================

def transform_skewed_features(df):
    """
    Apply log1p transformation to highly skewed
    numerical variables.
    """

    print("=" * 60)
    print("STEP 6: FEATURE TRANSFORMATION")
    print("=" * 60)

    df = df.copy()

    skewed_features = [

        "Deal_Dependency",
        "Average_Spending_per_Purchase",
        "Total_Campaign_Acceptance",
        "NumDealsPurchases",
        "NumCatalogPurchases",
        "NumWebPurchases",
        "MntFruits",
        "MntFishProducts",
        "MntSweetProducts",
        "MntMeatProducts",
        "MntGoldProds",
        "MntWines"
    ]

    transformed_features = []

    for column in skewed_features:

        if column in df.columns:

            # Make sure values are non-negative
            df[column] = df[column].clip(
                lower=0
            )

            df[column] = np.log1p(
                df[column]
            )

            transformed_features.append(
                column
            )

    print("Log transformation applied to:")

    for feature in transformed_features:
        print(" -", feature)

    print()

    return df


# ============================================================
# 7. BUILD PREPROCESSING PIPELINE
# ============================================================

def build_preprocessing_pipeline(df):
    """
    Build a reusable sklearn preprocessing pipeline.

    Numerical features:
        Missing value imputation
        StandardScaler

    Categorical features:
        Missing value imputation
        One-Hot Encoding
    """

    print("=" * 60)
    print("STEP 7: BUILDING PREPROCESSING PIPELINE")
    print("=" * 60)

    numerical_features = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = df.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    print(
        "Numerical features:",
        len(numerical_features)
    )

    print(
        "Categorical features:",
        len(categorical_features)
    )

    # --------------------------------------------------------
    # Numerical pipeline
    # --------------------------------------------------------

    numerical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # --------------------------------------------------------
    # Combined pipeline
    # --------------------------------------------------------

    transformers = []

    if len(numerical_features) > 0:

        transformers.append(
            (
                "numerical",
                numerical_pipeline,
                numerical_features
            )
        )

    if len(categorical_features) > 0:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    print("Preprocessing pipeline created successfully.")
    print()

    return preprocessor


# ============================================================
# 8. APPLY PIPELINE
# ============================================================

def apply_pipeline(
    df,
    preprocessor
):
    """
    Apply the preprocessing pipeline
    and return a DataFrame.
    """

    print("=" * 60)
    print("STEP 8: APPLYING PREPROCESSING PIPELINE")
    print("=" * 60)

    processed_array = (
        preprocessor.fit_transform(df)
    )

    # Get feature names after encoding
    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    processed_df = pd.DataFrame(
        processed_array,
        columns=feature_names,
        index=df.index
    )

    print(
        "Pipeline applied successfully."
    )

    print(
        "Final rows:",
        processed_df.shape[0]
    )

    print(
        "Final columns:",
        processed_df.shape[1]
    )

    print()

    return processed_df


# ============================================================
# 9. VALIDATE FINAL DATASET
# ============================================================

def validate_final_dataset(df):
    """
    Validate the final machine-learning-ready dataset.
    """

    print("=" * 60)
    print("STEP 9: FINAL DATASET VALIDATION")
    print("=" * 60)

    missing_values = (
        df.isnull()
        .sum()
        .sum()
    )

    duplicate_rows = (
        df.duplicated()
        .sum()
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    print(
        "Missing values:",
        missing_values
    )

    print(
        "Duplicate rows:",
        duplicate_rows
    )

    print(
        "Numerical features:",
        len(numeric_columns)
    )

    # Check scaling
    means = df[numeric_columns].mean()
    stds = df[numeric_columns].std()

    mean_check = np.allclose(
        means,
        0,
        atol=0.01
    )

    std_check = np.allclose(
        stds,
        1,
        atol=0.01
    )

    print(
        "Features approximately centered at 0:",
        mean_check
    )

    print(
        "Features approximately scaled to std=1:",
        std_check
    )

    print()

    return {
        "Missing Values": missing_values,
        "Duplicate Rows": duplicate_rows,
        "Mean Approximately 0": mean_check,
        "Std Approximately 1": std_check
    }


# ============================================================
# 10. COMPLETE PREPROCESSING PIPELINE
# ============================================================

def preprocessing_pipeline(input_file):

    print("\n")
    print("=" * 60)
    print("CUSTOMER PERSONALITY PREPROCESSING PIPELINE")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_data(
        input_file
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    df = remove_duplicates(
        df
    )

    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    df = handle_missing_values(
        df
    )

    # --------------------------------------------------------
    # Feature creation
    # --------------------------------------------------------

    df = create_features(
        df
    )

    # --------------------------------------------------------
    # Feature selection
    # --------------------------------------------------------

    df = select_features(
        df
    )

    # --------------------------------------------------------
    # Transform skewed numerical features
    # --------------------------------------------------------

    df = transform_skewed_features(
        df
    )

    # --------------------------------------------------------
    # Build pipeline
    # --------------------------------------------------------

    preprocessor = build_preprocessing_pipeline(
        df
    )

    # --------------------------------------------------------
    # Apply pipeline
    # --------------------------------------------------------

    processed_df = apply_pipeline(
        df,
        preprocessor
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validation_results = validate_final_dataset(
        processed_df
    )

    return (
        processed_df,
        preprocessor,
        validation_results
    )


# ============================================================
# 11. MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # FIND PROJECT ROOT
    # --------------------------------------------------------

    # preprocessing_pipeline.py is located at:
    #
    # Customer-Personality-Analysis/
    #     Feature_Engineering/
    #         Pipeline/
    #             preprocessing_pipeline.py
    #
    # Therefore, project root is two levels above this file.

    script_directory = os.path.dirname(
        os.path.abspath(__file__)
    )

    project_root = os.path.abspath(
        os.path.join(
            script_directory,
            "..",
            ".."
        )
    )

    # --------------------------------------------------------
    # INPUT FILE
    # --------------------------------------------------------

    input_file = os.path.join(
        project_root,
        "Cleaned_Data",
        "customer_personality_cleaned.csv"
    )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    output_directory = os.path.join(
        project_root,
        "Feature_Engineering",
        "Pipeline"
    )

    os.makedirs(
        output_directory,
        exist_ok=True
    )

    # --------------------------------------------------------
    # OUTPUT FILES
    # --------------------------------------------------------

    output_file = os.path.join(
        output_directory,
        "pipeline_output.csv"
    )

    pipeline_file = os.path.join(
        output_directory,
        "customer_preprocessing_pipeline.pkl"
    )

    # --------------------------------------------------------
    # DISPLAY PATHS
    # --------------------------------------------------------

    print("=" * 60)
    print("CUSTOMER PERSONALITY PREPROCESSING PIPELINE")
    print("=" * 60)

    print("\nProject root:")
    print(project_root)

    print("\nInput dataset:")
    print(input_file)

    # --------------------------------------------------------
    # CHECK INPUT FILE
    # --------------------------------------------------------

    if not os.path.exists(input_file):

        print("\nERROR: Input dataset not found.")

        print("\nExpected file:")
        print(input_file)

        raise FileNotFoundError(
            f"\nDataset not found:\n{input_file}"
        )

    print("\nInput dataset found successfully!")

    # --------------------------------------------------------
    # RUN PIPELINE
    # --------------------------------------------------------

    (
        final_df,
        preprocessor,
        validation_results
    ) = preprocessing_pipeline(
        input_file
    )

    # --------------------------------------------------------
    # SAVE FINAL DATASET
    # --------------------------------------------------------

    final_df.to_csv(
        output_file,
        index=False
    )

    # --------------------------------------------------------
    # SAVE REUSABLE PIPELINE
    # --------------------------------------------------------

    joblib.dump(
        preprocessor,
        pipeline_file
    )

    # --------------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nFinal dataset saved at:")
    print(output_file)

    print("\nReusable pipeline saved at:")
    print(pipeline_file)

    print("\nFinal dataset shape:")
    print(final_df.shape)

    print("\nValidation Results:")

    for key, value in validation_results.items():

        print(
            f"{key}: {value}"
        )

    print("\nFirst 5 rows of final dataset:")
    print(final_df.head())

    print(
        "\nTask 7 preprocessing pipeline finished successfully."
    )
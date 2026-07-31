import pandas as pd

# ==============================
# Load Dataset
# ==============================

input_file = "01_Raw_Data/marketing_campaign.csv"

df = pd.read_csv(input_file, sep="\t")

print("Dataset Loaded Successfully")
print(df.shape)

# ==============================
# Handle Missing Values
# ==============================

df["Income"] = df["Income"].fillna(df["Income"].median())

print("Missing values handled.")

# ==============================
# Remove Duplicate Records
# ==============================

df = df.drop_duplicates()

df = df.drop_duplicates(subset="ID")

print("Duplicates removed.")

# ==============================
# Correct Data Types
# ==============================

df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    format="%d-%m-%Y",
    errors="coerce"
)

print("Data types corrected.")

# ==============================
# Create Age Column
# ==============================

current_year = 2025

df["Age"] = current_year - df["Year_Birth"]

# Remove unrealistic ages

df = df[df["Age"] <= 100]

print("Age validated.")

# ==============================
# Standardize Categories
# ==============================

df["Education"] = df["Education"].str.strip().str.title()

df["Marital_Status"] = df["Marital_Status"].str.strip().str.title()

print("Categories standardized.")

# ==============================
# Handle Outliers (Winsorization)
# ==============================

columns = [
    "Income",
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

for col in columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df[col] = df[col].clip(lower, upper)

print("Outliers handled.")

# ==============================
# Save Cleaned Dataset
# ==============================

output_file = "Cleaned_Data/customer_personality_cleaned.csv"

df.to_csv(output_file, index=False)

print("Cleaned dataset exported successfully!")

print("\nPreprocessing Completed Successfully!")
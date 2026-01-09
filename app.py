
import streamlit as st 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder,StandardScaler,MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
@st.cache_data  # Cache data to avoid reloading on every Streamlit rerun
def load_data():
    return pd.read_csv("train.csv")

@st.cache_data
def get_clean_data():
    train = load_data()
    train["LotFrontage"] = train["LotFrontage"].fillna(train["LotFrontage"].median())
    train["MasVnrArea"] = train["MasVnrArea"].fillna(train["MasVnrArea"].median())
    train["GarageYrBlt"] = train["GarageYrBlt"].fillna(train["YearBuilt"])

    train["Electrical"] = train["Electrical"].fillna(train["Electrical"].mode()[0])

    fill_cols = {
        'MasVnrType': 'None',
        'BsmtQual': 'NoBasement',
        'BsmtCond': 'NoBasement',
        'BsmtExposure': 'NoBasement',
        'BsmtFinType1': 'NoBasement',
        'BsmtFinType2': 'NoBasement',
        'FireplaceQu': 'NoFireplace',
        'GarageType': 'NoGarage',
        'GarageFinish': 'NoGarage',
        'GarageQual': 'NoGarage',
        'GarageCond': 'NoGarage',
        'Fence': 'NoFence'
    }

    for col, val in fill_cols.items():
        train[col] = train[col].fillna(val)

    train.drop(columns=["Id", "PoolQC", "MiscFeature","Alley"], inplace=True)

    return train

col_des={"SalePrice" :"The property's sale price in dollars. This is the target variable that you're trying to predict",
"MSSubClass":"The building class",
"MSZoning":"The general zoning classification",
"LotFrontage":"Linear feet of street connected to property",
"LotArea":"Lot size in square feet",
"Street":"Type of road access",
"Alley":"Type of alley access",
"LotShape":"General shape of property",
"LandContour":"Flatness of the property",
"Utilities":"Type of utilities available",
"LotConfig":"Lot configuration",
"LandSlope":"Slope of property",
"Neighborhood":"Physical locations within Ames city limits",
"Condition1":"Proximity to main road or railroad",
"Condition2":"Proximity to main road or railroad (if a second is present)",
"BldgType":"Type of dwelling",
"HouseStyle":"Style of dwelling",
"OverallQual":"Overall material and finish quality",
"OverallCon":"Overall condition rating",
"YearBuilt":"Original construction date",
"YearRemodAdd":"Remodel date",
"RoofStyle":"Type of roof",
"RoofMatl":"Roof material",
"Exterior1st":"Exterior covering on house",
"Exterior2nd":"Exterior covering on house (if more than one material)",
"MasVnrType":"Masonry veneer type",
"MasVnrArea":"Masonry veneer area in square feet",
"ExterQual":"Exterior material quality",
"ExterCond":"Present condition of the material on the exterior",
"Foundation":"Type of foundation",
"BsmtQual":"Height of the basement",
"BsmtCond":"General condition of the basement",
"BsmtExposure":"Walkout or garden level basement walls",
"BsmtFinType1":"Quality of basement finished area",
"BsmtFinSF1":"Type 1 finished square feet",
"BsmtFinType2":"Quality of second finished area (if present)",
"BsmtFinSF2":"Type 2 finished square feet",
"BsmtUnfSF":"Unfinished square feet of basement area",
"TotalBsmtSF":"Total square feet of basement area",
"Heating":"Type of heating",
"HeatingQC":"Heating quality and condition",
"CentralAir":"Central air conditioning",
"Electrical":"Electrical system",
"1stFlrSF":"First Floor square feet",
"2ndFlrSF":"Second floor square feet",
"LowQualFinSF":"Low quality finished square feet (all floors)",
"GrLivArea":"Above grade (ground) living area square feet",
"BsmtFullBath":"Basement full bathrooms",
"BsmtHalfBath":"Basement half bathrooms",
"FullBath":"Full bathrooms above grade",
"HalfBath":"Half baths above grade",
"Bedroom":"Number of bedrooms above basement level",
"Kitchen":"Number of kitchens",
"KitchenQual":"Kitchen quality",
"TotRmsAbvGrd":"Total rooms above grade (does not include bathrooms)",
"Functional":"Home functionality rating",
"Fireplaces":"Number of fireplaces",
"FireplaceQu":"Fireplace quality",
"GarageType":"Garage location",
"GarageYrBlt":"Year garage was built",
"GarageFinish":"Interior finish of the garage",
"GarageCars":"Size of garage in car capacity",
"GarageArea":"Size of garage in square feet",
"GarageQual":"Garage quality",
"GarageCond":"Garage condition",
"PavedDrive":"Paved driveway",
"WoodDeckSF":"Wood deck area in square feet",
"OpenPorchSF":"Open porch area in square feet",
"EnclosedPorch":"Enclosed porch area in square feet",
"3SsnPorch": "Three season porch area in square feet",
"ScreenPorch":"Screen porch area in square feet",
"PoolArea": "Pool area in square feet",
"PoolQC": "Pool quality",
"Fence": "Fence quality",
"MiscFeature": "Miscellaneous feature not covered in other categories",
"MiscVal": "$Value of miscellaneous feature",
"MoSold": "Month Sold",
"YrSold": "Year Sold",
"SaleType": "Type of sale",
"SaleCondition": "Condition of sale"
}
st.set_page_config(page_title="House Prices project",layout="wide")
st.title("House Prices")
st.markdown("📌 Data Soure: "
            "[House Prices - Advanced Regression Techniques (Kaggle)]"
            "(https://www.kaggle.com/c/house-prices-advanced-regression-techniques) "
            )
st.image(
    "house.png",caption="House Price",use_container_width=True)

st.sidebar.title("Project Sections")
Section = st.sidebar.radio("Go to ", ["Overview","Load Data","Missing Values","Missing Handling","Encoding","Scaling","Visualizations"])

if Section == "Overview":
    st.header("📌 Project Overview")
    st.write("""
             This project focuses on analyzing house prices data.
             The main goal is to explore the dataset , handle missing values , 
             and prepare the data for future machine learning mpdels
             """)
    train = load_data()
    st.subheader("SalePrice Distibution")
    fig,ax=plt.subplots()
    sns.histplot(train["SalePrice"],kde=True,ax=ax)
    st.pyplot(fig)
    
elif Section =="Load Data":
    
    st.header("📁 Load Dataset")
    train = load_data()
    pre,info = st.tabs(
        ["📃 Data preview","ℹ️ Dataset Information"])
    with pre:
        st.subheader("Data preview")
        st.dataframe(train)
        st.write("Rows:",train.shape[0])
        st.write("Columns:",train.shape[1])
        
        
    with info:
        st.subheader("Data Information")
        stat,meta=st.tabs([" 📉 Data Types","ℹ️ MetaData"])
        with stat:
           st.subheader("Dataset Information")
           st.dataframe(train.dtypes.reset_index().rename(columns={"index": "Column", 0: "Data Type"}))

        with meta:
            st.subheader("Columns Description:")
            select_cols = st.selectbox(
                "Select a column",train.columns)
            if select_cols in col_des:
                st.info(col_des[select_cols])
            else:
                st.caption("No description available for this column")
       
elif Section == "Missing Values":
    st.header("❓ Missing Values Analysis")
    over,num,categ = st.tabs(["📊 Overview"," 🔢 Numeric Columns"," 🔠 Categorical Columns"])
    train = load_data()

    with over:
        st.subheader("Overall Missing Values")
        total_missing = train.isna().sum().sum()
        st.write("Total missing values in dataset:",total_missing)
        missing_columns =train.isna().sum()
        missing_df =   missing_columns [  missing_columns  > 0].sort_values(ascending=False)
        st.dataframe(missing_df)
        st.caption("Features with missing values")
    with num:
        st.subheader("Missing Values in Numeric Columns: ")
        num_cols=train.select_dtypes(include=["int64","float64"]).columns
        num_missing=train[num_cols].isna().sum()
        st.dataframe(num_missing[num_missing>0])
    with categ:
        st.subheader("Missing Values in Categorical Columns: ")
        categ_cols =train.select_dtypes(include="object").columns
        categ_mis=train[categ_cols].isna().sum()
        st.dataframe(categ_mis[categ_mis>0])
        st.subheader("Value Counts for Categorical Feature")
        select_cat=st.selectbox("Select a  categorical column",categ_cols)
        st.write(train[select_cat].value_counts())
        
elif Section =="Missing Handling":
    st.header("🧹 Missing Handling Values")
    raw = load_data()
    clean = get_clean_data()
    st.write("Before Handling,total missing values:",raw.isna().sum().sum())
    st.write("After Handling,total missing values after handling:",clean.isna().sum().sum())
    st.success("Missing values handled successfully ✅")
    
    
elif Section =="Encoding":
    st.header("🔤 Data Encoding")
    train = get_clean_data()
    categ_cols = train.select_dtypes(include="object").columns
    encoded=pd.get_dummies(train,columns=categ_cols,drop_first=True)
    st.write("Columns before encoding:",train.shape[1])
    st.write("Columns after encoding:",encoded.shape[1])
    st.dataframe(encoded.astype(int).head())
    st.caption("One_hot encoding applied to categorical features")
    
elif Section =="Scaling":
    st.header("📏 Feature Scaling")
    train = get_clean_data()
    train = pd.get_dummies(train, drop_first=True)
    num_cols = train.select_dtypes(include=["int64","float64"]).columns
    num_cols = num_cols.drop("SalePrice")
    std, minmax = st.tabs([" 🎯 StandardScaler"," ↕ MinMaxScaler"])

    with std:
        st.subheader("Standard Scaling")
        scaler_std = StandardScaler()
        train_std = train.copy()
        train_std[num_cols] = scaler_std.fit_transform(train[num_cols])
        st.caption("StandardScaler applied: centered at =0 , unit variance")
        st.dataframe(train_std.round(2))

    with minmax:
        st.subheader("Min-Max Scaling")
        scaler_mm = MinMaxScaler()
        train_mm = train.copy()
        train_mm[num_cols] = scaler_mm.fit_transform(train[num_cols])
        st.caption("MinMaxScaler applied:scaled between 0 and  1")
        st.dataframe(train_mm.round(2))
        
elif Section =="Visualizations":
    st.header("📊 Data Visualizations")
    raw=load_data()
    clean=get_clean_data()
    st.subheader("Before Preprocessing")
    if st.checkbox("Show Histogram Before Preprocessing"):
        fig, ax = plt.subplots()
        ax.hist(raw["SalePrice"], bins=30, color="#69b3a2")
        ax.set_xlabel("SalePrice")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        st.caption("Histogram before handling missing values or encoding.")

    if st.checkbox("Show Boxplot Before Preprocessing"):
        fig, ax = plt.subplots()
        sns.boxplot(y=raw["SalePrice"], ax=ax)
        ax.set_ylabel("SalePrice")
        st.pyplot(fig)
        st.caption("Boxplot showing outliers before preprocessing.")
    
    if st.checkbox("Show Missing Values Bar Chart Before Preprocessing"):
        missing = raw.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.barh(missing.index, missing.values, color="#ff9999")
        ax.set_xlabel("Number of Missing Values")
        st.pyplot(fig)
        st.caption("Missing values per feature before preprocessing.")
    
    st.subheader("After Preprocessing")
    clean=get_clean_data()
    if st.checkbox("Show Histogram After Preprocessing"):
        fig, ax = plt.subplots()
        ax.hist(clean["SalePrice"], bins=30, color="#69b3a2")
        ax.set_xlabel("SalePrice")
        ax.set_ylabel("Count")
        st.pyplot(fig)
        st.caption("Histogram showing SalePrice distribution after preprocessing.")
    
    if st.checkbox("Show Boxplot After Preprocessing"):
        fig, ax = plt.subplots()
        sns.boxplot(y=clean["SalePrice"], ax=ax)
        ax.set_ylabel("SalePrice")
        st.pyplot(fig)
        st.caption("Boxplot highlighting high-value houses after preprocessing.")



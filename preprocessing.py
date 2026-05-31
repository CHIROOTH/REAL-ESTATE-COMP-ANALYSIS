import pandas as pd

# read the cleaned canada csv file
df = pd.read_csv('canada_housing_data.csv')
# get the total number of rows before cleaning duplicates
total_before_cleaning_duplicates = df.shape[0]
# get the total number of duplicates
duplicates = df.duplicated().sum()
# drop the duplicates
df_no_duplicates = df.drop_duplicates(keep='first')
# get the total number of rows after cleaning duplicates
total_after_cleaning_duplicates = df_no_duplicates.shape[0]
# get the total number of duplicates after cleaning
duplicates_after_cleaning = df_no_duplicates.duplicated().sum()

print(f"Total before cleaning duplicates: {total_before_cleaning_duplicates}")
print(f"Total duplicates: {duplicates}")
print(f"Total after cleaning duplicates: {total_after_cleaning_duplicates}")
print(f"Total duplicates after cleaning: {duplicates_after_cleaning}")
# get the columns with missing values
cols_with_missing_values = df_no_duplicates.columns[df_no_duplicates.isnull().any()].tolist()
print(f"Columns with missing values: {cols_with_missing_values}")
# get the rows with missing values
rows_with_missing_values = df_no_duplicates.isnull().any(axis=1).sum()
print(f"Rows with missing values: {rows_with_missing_values}")
# drop the rows with missing col values (too low hence we remove the columns itself)
df_cleaned = df_no_duplicates.dropna(how='any')
total_after_cleaning_missing_values = df_cleaned.shape[0]
print(f"Total after cleaning missing values: {total_after_cleaning_missing_values}")
# drop the columns with missing values
df_no_missing_values = df_no_duplicates.drop(columns=cols_with_missing_values, errors='ignore')
total_after_cleaning_missing_values_columns = df_no_missing_values.shape[0]
print(f"Total after cleaning missing values columns: {total_after_cleaning_missing_values_columns}")
# save the cleaned dataframe to a csv file
df_no_missing_values.to_csv('cleaned_canada_housing_data.csv', index=False)
# df_cleaned.to_csv('cleaned_canada_housing_data.csv', index=False)

print("remaining columns: ", df_no_missing_values.columns.tolist())
print("shape after removing missing values: ", df_no_missing_values.shape)
# print("shape after removing missing values: ", df_cleaned.shape)

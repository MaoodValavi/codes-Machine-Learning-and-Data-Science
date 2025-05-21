import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# بارگذاری داده‌ها
file_path=r"C:\Users\masud\Personallaptob12Jan2025\Freelancer\Parscoders\7_amazon\amazon_sales_data 2025.csv"

df = pd.read_csv(file_path)

# تبدیل ستون تاریخ به فرمت datetime
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y')

# حذف ستون‌های غیرضروری
df_model = df.drop(columns=['Order ID', 'Customer Name'])

# استخراج ویژگی‌های زمانی از ستون تاریخ
df_model['Year'] = df['Date'].dt.year
df_model['Month'] = df['Date'].dt.month
df_model['Day'] = df['Date'].dt.day
df_model = df_model.drop(columns=['Date'])

# کدگذاری متغیرهای دسته‌ای (categorical)
df_model = pd.get_dummies(df_model, drop_first=True)

# تعریف ویژگی‌ها و هدف
X = df_model.drop('Price', axis=1)
y = df_model['Price']

# تقسیم داده‌ها به آموزش و تست
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# آموزش مدل
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# پیش‌بینی و ارزیابی مدل
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

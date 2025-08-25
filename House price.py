import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
np.random.seed(42)
sqft = np.random.randint(500, 3000, 100).reshape(-1,1)
price = 30000 + 120 * sqft + np.random.randint(-20000,20000,100).reshape(-1,1)
model = LinearRegression().fit(sqft, price)
pred = model.predict(sqft)
print("Intercept (β0):", model.intercept_[0])
print("Slope (β1):", model.coef_[0][0])
print("R² Score:", r2_score(price, pred))
print("MAE:", mean_absolute_error(price, pred))
plt.scatter(sqft, price, color="blue", label="Data points")
plt.plot(sqft, pred, color="red", linewidth=2, label="Regression line")
plt.xlabel("Square Footage")
plt.ylabel("House Price (USD)")
plt.title("Linear Regression: House Price vs. Square Footage")
plt.legend()
plt.show()
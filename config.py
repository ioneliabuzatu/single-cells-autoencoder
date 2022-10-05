from sklearn.preprocessing import MinMaxScaler, StandardScaler

latent_space = 100
batch_size = 500
lr = 0.01

norms_types = {"minmax": lambda x: MinMaxScaler().fit_transform(x), "standardization": None}
norm = "nope"
# norm = "minmax"
# norm = "standardization" 

# Scale data to have zero mean and unit variance
# scaler = StandardScaler()
# scaler.fit(X_train)
# x_train = scaler.transform(X_train)
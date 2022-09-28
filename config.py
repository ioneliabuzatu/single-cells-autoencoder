from sklearn.preprocessing import MinMaxScaler

batch_size = 10

norms_types = {"minmax": lambda x: MinMaxScaler().fit_transform(x)}
norm = "nope"
# norm = "minmax"
# norm = "standardization" 
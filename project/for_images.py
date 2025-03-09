import optuna
import optuna.visualization as vis
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt

# Eğitilmiş modeli yükleyin
model = load_model('trained_model.h5')

# Optuna çalışma alanını yükleyin
study = optuna.load_study(study_name="time_series_optimization_gpu", storage='sqlite:///optuna_study_gpu.db')

# Optimizasyon geçmişini görselleştirin
vis.plot_optimization_history(study).show()

# Parametrelerin önemini görselleştirin
vis.plot_param_importances(study).show()

# Hiperparametrelerin dağılımını görselleştirin
vis.plot_parallel_coordinate(study).show()

# Modelin yapısını görselleştirin ve kaydedin
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

# Modelin yapısını gösterin
img = plt.imread('model.png')
plt.imshow(img)
plt.axis('off')
plt.show()


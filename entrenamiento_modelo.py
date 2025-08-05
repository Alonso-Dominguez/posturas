import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class GestureModelTraining:
    def __init__(self, dataset_file="gesture_dataset.csv"):
        self.dataset_file = dataset_file
        self.model = None
        self.scaler = StandardScaler()
        
        self.gestos = {
            0: "cerrado",
            1: "abierto", 
            2: "pulgar_arriba",
            3: "paz",
            4: "apuntar"
        }
    
    def load_dataset(self):
        try:
            df = pd.read_csv(self.dataset_file)
            print(f"Dataset cargado: {len(df)} muestras")
            print(f"Columnas: {df.shape[1]} (42 coordenadas + 1 estado)")
            
            print("\nDistribución de gestos:")
            for gesture_id, count in df['estado'].value_counts().sort_index().items():
                gesture_name = self.gestos.get(gesture_id, f"Gesto_{gesture_id}")
                print(f"  {gesture_name}: {count} muestras")
            
            return df
            
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo {self.dataset_file}")
            print("Primero ejecuta el script de captura de dataset")
            return None
        except Exception as e:
            print(f"Error al cargar dataset: {e}")
            return None
    
    def preprocess_data(self, df):
        X = df.drop('estado', axis=1)
        y = df['estado']
        
        if X.isnull().any().any():
            print("Advertencia: Se encontraron valores NaN en los datos")
            X = X.fillna(0)
        
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y
    
    def train_models(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nDatos de entrenamiento: {len(X_train)} muestras")
        print(f"Datos de prueba: {len(X_test)} muestras")
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n--- Entrenando {name} ---")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Precisión: {accuracy:.4f}")
            
            max_probas = np.max(y_proba, axis=1)
            print(f"Confianza promedio: {np.mean(max_probas):.4f}")
            print(f"Confianza mínima: {np.min(max_probas):.4f}")
            print(f"Confianza máxima: {np.max(max_probas):.4f}")
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'y_test': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba
            }
        
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        self.model = results[best_model_name]['model']
        
        print(f"\n=== MEJOR MODELO: {best_model_name} ===")
        print(f"Precisión: {results[best_model_name]['accuracy']:.4f}")
        
        self.test_confidence_thresholds(results[best_model_name])
        
        print("\nReporte de clasificación:")
        target_names = [self.gestos[i] for i in sorted(self.gestos.keys())]
        print(classification_report(
            results[best_model_name]['y_test'], 
            results[best_model_name]['y_pred'],
            target_names=target_names if len(set(y)) == len(target_names) else None
        ))
        
        return results[best_model_name]
    
    def test_confidence_thresholds(self, result):
        print("\n=== ANÁLISIS DE UMBRALES DE CONFIANZA ===")
        
        y_test = result['y_test']
        y_pred = result['y_pred'] 
        y_proba = result['y_proba']
        max_probas = np.max(y_proba, axis=1)
        
        thresholds = [0.7, 0.8, 0.85, 0.9, 0.95]
        
        for threshold in thresholds:
            confident_mask = max_probas >= threshold
            if np.sum(confident_mask) > 0:
                y_test_conf = y_test[confident_mask]
                y_pred_conf = y_pred[confident_mask]
                accuracy_conf = accuracy_score(y_test_conf, y_pred_conf)
                coverage = np.sum(confident_mask) / len(y_test)
                rejected = 1 - coverage
                
                print(f"Umbral {threshold:.0%}: Precisión={accuracy_conf:.3f}, "
                      f"Cobertura={coverage:.1%}, Rechazadas={rejected:.1%}")
            else:
                print(f"Umbral {threshold:.0%}: Sin predicciones confiables")
        
        print("\nRecomendación: Usar umbral de 90% para mejor balance")
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Visualiza la matriz de confusión"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        labels = [self.gestos.get(i, f'Gesto_{i}') for i in sorted(set(y_test))]

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Valor Real')
        plt.tight_layout()
        plt.show()
    
    def predict_with_confidence(self, X, threshold=0.9):
        if self.model is None:
            print("Error: No hay modelo entrenado")
            return None, None
        
        probabilities = self.model.predict_proba(X)
        max_proba = np.max(probabilities, axis=1)
        predictions = self.model.predict(X)
        
        confident_predictions = []
        confidences = []
        
        for i, (pred, conf) in enumerate(zip(predictions, max_proba)):
            if conf >= threshold:
                confident_predictions.append(pred)
                confidences.append(conf)
            else:
                confident_predictions.append(-1)
                confidences.append(conf)
        
        return confident_predictions, confidences
    
    def save_model(self, model_name="gesture_model.pkl", scaler_name="gesture_scaler.pkl"):
        if self.model is None:
            print("Error: No hay modelo entrenado para guardar")
            return
        
        joblib.dump(self.model, model_name)
        print(f"Modelo guardado como: {model_name}")
        
        joblib.dump(self.scaler, scaler_name)
        print(f"Escalador guardado como: {scaler_name}")
    
    def test_prediction(self, X_test, y_test):
        if self.model is None:
            print("Error: No hay modelo entrenado")
            return
        
        print("\n=== PRUEBAS DE PREDICCIÓN ===")
        for i in range(min(5, len(X_test))):
            sample = X_test[i:i+1]
            prediction = self.model.predict(sample)[0]
            actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
            
            pred_name = self.gestos.get(prediction, f"Gesto_{prediction}")
            actual_name = self.gestos.get(actual, f"Gesto_{actual}")
            status = "✓" if prediction == actual else "✗"
            print(f"Muestra {i+1}: Predicho={pred_name}, Real={actual_name} {status}")

def main():
    trainer = GestureModelTraining()
    
    df = trainer.load_dataset()
    if df is None or len(df) < 10:
        print("Error: Dataset no válido o insuficiente")
        return
    
    X, y = trainer.preprocess_data(df)
    best_result = trainer.train_models(X, y)
    
    # Nueva función corregida
    trainer.plot_confusion_matrix(best_result['y_test'], best_result['y_pred'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    trainer.test_prediction(X_test, y_test)
    trainer.save_model()
    
    print("\n=== ENTRENAMIENTO COMPLETADO ===")
    print("Archivos generados:")
    print("- gesture_model.pkl (modelo entrenado)")
    print("- gesture_scaler.pkl (escalador)")

if __name__ == "__main__":
    main()

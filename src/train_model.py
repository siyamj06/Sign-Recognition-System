import os
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    data = np.array(d['data'])
    labels = np.array(d['labels'])
    print(f"📊 Loaded: {data.shape[0]} samples, {data.shape[1]} features")
    print(f"   Classes: {sorted(np.unique(labels))}")
    return data, labels


def train_and_evaluate(train_path, test_path, val_path, model_path, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)

    # 🔹 Load datasets
    print("\n🔹 Loading TRAIN dataset (person_1)")
    X_train, y_train = load_data(train_path)

    print("\n🔹 Loading TEST dataset (person_2)")
    X_test, y_test = load_data(test_path)

    print("\n🔹 Loading VALIDATION dataset (person_3)")
    X_val, y_val = load_data(val_path)

    # 🔹 Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    y_val = le.transform(y_val)

    class_names = le.classes_

    # FEATURE SCALING (VERY IMPORTANT)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    print(f"\n   Train: {len(X_train)} | Test: {len(X_test)} | Val: {len(X_val)}")

    # Class distribution plot
    unique, counts = np.unique(le.inverse_transform(y_train), return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(unique, counts, color='#4CAF50', edgecolor='black')
    plt.title('Training Data Distribution')
    plt.savefig(os.path.join(plots_dir, 'class_distribution.png'))
    plt.close()

    # Models
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=500, max_depth=25, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', class_weight='balanced', random_state=42, n_jobs=-1),
        'SVM': SVC(
            kernel='rbf', C=2, gamma='scale', probability=True),
        'KNN': KNeighborsClassifier(
            n_neighbors=7, weights='distance', n_jobs=-1),
        'MLP': MLPClassifier(
            hidden_layer_sizes=(128,64), max_iter=300, alpha=0.0005, random_state=42, early_stopping=True, validation_fraction=0.1)
    }

    results = {}
    print("\n" + "=" * 60)
    print("🔬 TRAINING MODELS")
    print("=" * 60)

    for name, model in models.items():
        print(f"\n🔄 {name}...")
        model.fit(X_train, y_train)
        # Evaluate
        y_test_pred = model.predict(X_test)
        y_val_pred = model.predict(X_val)

        test_acc = accuracy_score(y_test, y_test_pred)
        val_acc = accuracy_score(y_val, y_val_pred)
        cv = cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)

        results[name] = {
            'model': model,
            'test_acc': test_acc,
            'val_acc': val_acc,
            'cv_mean': cv.mean(),
            'y_val_pred': y_val_pred
        }
        print(f"   ✅ Test: {test_acc*100:.2f}% | Val: {val_acc*100:.2f}% | CV: {cv.mean()*100:.2f}%")

    # Comparison plot
    names = list(results.keys())
    test_scores = [results[n]['test_acc']*100 for n in names]
    val_scores = [results[n]['val_acc']*100 for n in names]

    x = np.arange(len(names))
    w = 0.35

    plt.figure(figsize=(10,6))
    plt.bar(x - w/2, test_scores, w, label='Test')
    plt.bar(x + w/2, val_scores, w, label='Validation')
    plt.xticks(x, names)
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.title("Model Comparison")
    plt.savefig(os.path.join(plots_dir, 'model_comparison.png'))
    plt.close()

    # Best model
    best_name = max(results, key=lambda x: results[x]['val_acc'])
    best = results[best_name]

    print("\n" + "=" * 60)
    print(f"🏆 BEST MODEL: {best_name}")
    print(f"   Test Accuracy: {best['test_acc']*100:.2f}%")
    print(f"   Validation Accuracy: {best['val_acc']*100:.2f}%")
    print("=" * 60)

    # Decode labels back for report
    y_val_decoded = le.inverse_transform(y_val)
    y_pred_decoded = le.inverse_transform(best['y_val_pred'])

    print(f"\n📋 Classification Report:")
    print(classification_report(y_val_decoded, y_pred_decoded))

    # Confusion matrix
    cm = confusion_matrix(y_val_decoded, y_pred_decoded, labels=class_names)

    plt.figure(figsize=(12,10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.close()

    # Save
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': best['model'],
            'scaler': scaler,
            'label_encoder': le,
            'model_name': best_name,
            'accuracy': best['val_acc']
        }, f)

    print(f"\n💾 Model saved: {model_path}")
    print(f"📊 Plots saved: {plots_dir}")


if __name__ == "__main__":
    TRAIN_PATH = os.path.join("..", "data", "processed", "train.pkl")
    TEST_PATH  = os.path.join("..", "data", "processed", "test.pkl")
    VAL_PATH   = os.path.join("..", "data", "processed", "val.pkl")

    MODEL_PATH = os.path.join("..", "models", "sign_language_model.pkl")
    PLOTS_DIR  = os.path.join("..", "plots")

    train_and_evaluate(TRAIN_PATH, TEST_PATH, VAL_PATH, MODEL_PATH, PLOTS_DIR)
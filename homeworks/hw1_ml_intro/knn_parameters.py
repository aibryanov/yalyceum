from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
import pandas as pd


if __name__ == "__main__":
    
    data = pd.read_csv("penguins.csv")
    data.dropna(inplace=True)
    labels = data["species"].copy()
    features = data[["bill_length_mm", "bill_depth_mm"]].copy()
    
    train_features, test_features, train_labels, test_labels = train_test_split( 
                features, labels, test_size=0.2, random_state=123)
    
    weights = ["uniform", "distance"]
    best_acc = 0
    worst_acc = 1
    best_params = ""
    worst_params = ""
    
    for weight in weights:
        for i in range(1, 11):
            knn = KNeighborsClassifier(n_neighbors=i, weights=weight)
            knn.fit(train_features, train_labels)
            y_pred = knn.predict(test_features)
            
            accuracy = metrics.accuracy_score(test_labels, y_pred)
            if accuracy < worst_acc:
                worst_acc = accuracy
                worst_params = (i, weight)
                
            if accuracy > best_acc:
                best_acc = accuracy
                best_params = (i, weight)
                
    print("Best accuracy: {:.6f}".format(best_acc)) 
    print("Worst accuracy: {:.6f}".format(worst_acc))
    
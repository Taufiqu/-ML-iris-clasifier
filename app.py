import gradio as gr
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
model = RandomForestClassifier()
model.fit(iris.data, iris.target)

def predict(sepal_length, sepal_width, petal_length, petal_width):
    prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
    return iris.target_names[prediction][0]

inputs = [
    gr.Slider(4.0, 8.0, label="Sepal Length"),
    gr.Slider(2.0, 4.5, label="Sepal Width"),
    gr.Slider(1.0, 7.0, label="Petal Length"),
    gr.Slider(0.1, 2.5, label="Petal Width")
]

gr.Interface(fn=predict, inputs=inputs, outputs="text", title="Iris Classifier").launch()

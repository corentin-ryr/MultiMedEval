import streamlit as st
import pandas as pd

import csv


@st.cache_data
def load_data():
    with open('MedQA.csv', "r", newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        data = []
        for row in reader:
            data.append(row)
    return data


# On click, update the current index and the fields "correctAnswerField" and "modelsAnswer".
# Save the answer in a csv file.
def on_click(humanAnswer:bool):
    currentIdx = st.session_state.currentIdx
    with open('MedQAEvaluated.csv', "a", newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow([data[currentIdx][0], data[currentIdx][1], data[currentIdx][2], humanAnswer])
    st.session_state.currentIdx += 1


if 'currentIdx' not in st.session_state:
    st.session_state.currentIdx = 0

data = load_data()

currentIdx = st.session_state.currentIdx
st.write(f"Current index: {currentIdx}")


left_column, right_column = st.columns(2)
with left_column:
    st.write("Correct answer")
    st.write(data[currentIdx][0].upper())

with right_column:
    st.write("Model's answer")
    modelsAnswer = st.write(data[currentIdx][1])

st.divider()
left_column, right_column = st.columns(2)
with left_column:
    st.write("The model's answer is correct")
    st.button("True", on_click=on_click, args=[True])
with right_column:
    st.write("The model's answer is wrong")
    st.button("False", on_click=on_click, args=[False])




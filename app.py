import streamlit as st
import spacy_streamlit
import spacy
# import joblib
# import pickle
from PyPDF2 import PdfReader
#from NERDA.models import NERDA
from spacy.tokens import Doc

import json
import pandas as pd
import numpy as np
from tabulate import tabulate
from azure.core.exceptions import ResourceNotFoundError
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import FormRecognizerClient, FormTrainingClient

nlp = spacy.load('model-last')

credentials = json.load(open('credential.json'))
API_KEY = credentials['API_KEY']
ENDPOINT = credentials['ENDPOINT']
form_recognizer_client = FormRecognizerClient(ENDPOINT, AzureKeyCredential(API_KEY))

def get_contents(img_obj):
    #with open('azure_test.png', "rb") as f:
    with open(img_obj, "rb") as f:
        poller = form_recognizer_client.begin_recognize_content(form=f)
    form_result = poller.result()

    contents, temp = [], []
    col_count, row_count = 0, 0
    cur_y, cur_x = 0, 0
    for page in form_result:
        for line in page.lines:
            #print('Column Count: {0}'.format(line.column_count))
            #print('Row Count: {0}'.format(line.row_count))
            print(line.text)
            temp.append(line.text)
            for cell in line.words:
                if abs(cell.bounding_box[0].y - cur_y) > 5:
                    cur_y = cell.bounding_box[0].y
                    row_count += 1
                print('Cell Value: {0}'.format(cell.text))
                print('Location: {0}'.format(cell.bounding_box))

    col_count = len(temp) // row_count
    cnt, t = 0, []
    for i in range(len(temp)):
        t.append(temp[i])
        if (i + 1) % col_count == 0:
            contents.append(t.copy())
            t = [] 

    return contents

def get_table_str(contents):
    head = contents[0]
    s = tabulate(contents[1:], headers=head, tablefmt="grid")
    return s

# with open('nerda-model.pickle','rb') as fr:
#     model = pickle.load(fr)

def main():
    st.title("Named Entity Recognizer For Fund Setup")
    st.subheader('This web application shows the pipline of extracting key information from fund prospectus.')

    st.sidebar.subheader("NT-Uchicago Capstone Showcase")
    #st.sidebar.markdown("This web application shows the pipline of extracting key information from fund prospectus.")

    st.sidebar.markdown("**Data Source**")
    menu = ['Input Text', 'PDF File']
    choice = st.sidebar.selectbox('Choose a data format', menu)

    colors = {"Fund name": "#AB8BEE",
                "Fund manager": "#54FBDD",
                "Calendar": "#FFD069",
                "Account base currency": "#F775E9",
                "Valuation point": "#D0FF97"}
    options = {"ents": ["Fund name"], "colors": colors}
    #displacy.serve(doc, style="ent", options=options)

    if choice == 'Input Text':
        st.subheader('Text Processor')
        raw_text = st.text_area("", "Enter Text Here")
        docx = nlp(raw_text)
        spacy_streamlit.visualize_ner(docx, labels=['Fund name',
              'Account base currency', 
              'Valuation point', 
              'Calendar', 
              'Fund manager'], colors=colors)

    elif choice == 'PDF File':
        uploaded_file = st.sidebar.file_uploader('Choose your .pdf file', type="pdf")

        st.sidebar.markdown("**Final Output**")

        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        #st.subheader('Text Processor')

        if uploaded_file is None:
            raw_text = st.text_area("", "Enter Text Here")
            docx = nlp(raw_text)
            spacy_streamlit.visualize_ner(docx, labels=['Fund name',
                'Account base currency', 
                'Valuation point', 
                'Calendar', 
                'Fund manager'], colors=colors)

        elif uploaded_file is not None:
            reader = PdfReader(uploaded_file)
            page = reader.pages[0]
            raw_text = page.extract_text()
            docx = nlp(raw_text)
            spacy_streamlit.visualize_ner(docx, labels=['Fund name',
              'Account base currency', 
              'Valuation point', 
              'Calendar', 
              'Fund manager'], colors=colors)

            #st.subheader("Table Processor")
            count = 0
            for image_file_object in page.images:
                with open(str(count) + image_file_object.name, "wb") as fp:
                    fp.write(image_file_object.data)
                    count += 1
                #with open(str(count - 1) + image_file_object.name, "rb") as fp:
                contents = get_contents(str(count - 1) + image_file_object.name)
                table_df = pd.DataFrame(contents[1:], columns=contents[0])
                st.text(get_table_str(contents))
            
            df_columns = ["Fund name", "Account base currency", "Valuation point", "Calendar", "Fund manager"]
            overall_contents = []
            
            dic = {"Fund name": float("nan"), "Account base currency": float("nan"), "Valuation point": float("nan"), "Calendar": float("nan"), "Fund manager": float("nan")}
            for ent in docx.ents:
                if ent.label_ == 'Fund name':
                    if dic['Fund name'] != float("nan"):
                        #overall_df.append(pd.DataFrame([dic]))
                        overall_contents.append(list(dic.values()))
                    dic = {"Fund name": float("nan"), "Account base currency": float("nan"), "Valuation point": float("nan"), "Calendar": float("nan"), "Fund manager": float("nan")}
                dic[ent.label_] = ent.text
                print(dic)
            if dic['Fund name'] != float("nan"):
                overall_contents.append(list(dic.values()))

            overall_df = pd.DataFrame(overall_contents, columns = df_columns)
            overall_df = pd.concat([overall_df, table_df], ignore_index = True)

            col_to_drop = []
            for col in overall_df.columns:
                if col not in ['Fund name', 'Account base currency', 'Valuation point', 'Calendar', 'Fund manager']:
                    col_to_drop.append(col)
            overall_df.drop(columns = col_to_drop, inplace = True)
            overall_df.drop(index = 0, inplace = True)
            
            csv = convert_df(overall_df)

            select = st.sidebar.checkbox("Show", False)
            st.sidebar.download_button(
                label="Download as CSV",
                data=csv,
                file_name='final_output.csv',
                mime='saved/csv',
            )

            if select == True:
                st.subheader("Final Output")
                st.markdown("Collects name entities extracted from both texts and tables.")
                st.dataframe(overall_df)
                

if __name__ == '__main__':
    main()
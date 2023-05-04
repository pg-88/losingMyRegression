import streamlit as st 
import pandas as pd
from modules.model import Model


def main():
    my_file = st.file_uploader("File csv:", accept_multiple_files=False, type=["csv"])
    st.write(my_file)
    st.write(Model(my_file).get_col_names())



if __name__ == '__main__':
    main()
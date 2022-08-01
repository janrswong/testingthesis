import streamlit as st

st.title('About the Research')

# a. how the thing works
st.header('How It Works')
st.markdown('This project uses Streamlit simply as a means to present findings. The real magic happens in Google Colab!')

# a.1 How to use the explore tab
st.subheader("How to use the 'Explore' tab")

#a.2 How to use the Make a model tab
st.subheader("How to use the 'Make a Model' tab")

# b. snippets of the paper
st.header('Documentation')
# sample text
st.markdown('Here are some code snippets!')

# sample add code snippet
code = '''def hello():
     print("Hello, Streamlit!")'''
st.code(code, language='python')

# sample add latex or formulas
st.latex(r'''
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     ''')

# c. about us creators
st.header('About the Authors')
st.markdown('The authors, Janna Rizza Wong, Stephanie Lorraine Ignas, and Alyanna Angela Castillon, are all university seniors taking Computer Science.')

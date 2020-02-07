import streamlit as st
from PIL import Image
#ML Packages..
import joblib

#Vectorizer
gender_vectorizer = open("Models/gender_vectorizer.pkl","rb")
gender_cv = joblib.load(gender_vectorizer)

#Models
gender_nv_model = open("Models/naivebayesgendermodel.pkl","rb")
gender_clf = joblib.load(gender_nv_model)



@st.cache
def predict_gender(data):
	vect = gender_cv.transform(data).toarray()
	result = gender_clf.predict(vect)
	return result


def load_images(image_name):
	img = Image.open(image_name)
	return st.image(img,width=300)





def main():
	"""Gender classifier App"""

	st.title("Gender Classifier Machine Learning App")
	st.subheader("Naive Bayes")

	html_temp ="""
	<div style="background-color:tomato; padding:20px;">
	<h2>Streamlit Machine Learning App</h2>



	<div style="padding-bottom:50px;">
	"""

	st.markdown(html_temp, unsafe_allow_html=True)

	name = st.text_input("Enter Name","")
	if st.button("Classify"):
		st.text("Name :{}".format(name.title()))
		result = predict_gender([name])
		if result[0] == 0:
			prediction = "Female"
			c_image = 'female.png'
		else:
			prediction = "Male"
			c_image = 'male.png'

		st.success("Name {} , is Classied as {}".format(name.title(),prediction))
		load_images(c_image)






if __name__ == '__main__':
		main()
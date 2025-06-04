from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from fastapi import APIRouter, HTTPException
import pyttsx3
import os

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_BUILD_DIR = os.path.join(BASE_DIR, "../frontend/build")

if os.path.exists(FRONTEND_BUILD_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_BUILD_DIR, html=True), name="static")

    @app.get("/")
    async def serve_react_app():
        return FileResponse(os.path.join(FRONTEND_BUILD_DIR, "index.html"))
else:
    print("⚠️ React build directory not found. Did you run `npm run build` in frontend?")


origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = tf.keras.models.load_model("../models/1")

CLASS_NAMES = ['Algal Tea',
 'Anthracnose Tea',
 'Bacterial blight cotton',
 'Bacterial spot tomato',
 'Black mold tomato',
 'Brown blight Tea',
 'Brown spot Rice',
 'Curl virus cotton',
 'Gray spot tomato',
 'Healthy Rice',
 'Healthy Tea',
 'Healthy cotton',
 'Late blight tomato',
 'cordana_Banana',
 'fussarium wilt cotton',
 'health tomato',
 'healthy_Banana',
 'leaf blast Rice',
 'leaf scald Rice',
 'narrow brown spot Rice',
 'pestalotiopsis_Banana',
 'red leaf spot Tea',
 'sigatoka_Banana',
 'white spot Tea'
]



def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)).resize((256,256)))
    return image



@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    
    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    predicted_class = predicted_class
    confidence = np.max(predictions[0])
    dict = {    
 'Algal Tea' :"use Soluble Fertilizers",
 'Anthracnose Tea':"use copper sprays",
 'Bacterial blight cotton':"use Copper-based fungicides",
 'Bacterial spot tomato':"use copper sprays & pathogen-free seed transplants",
 'Black mold tomato':"use copper sprays",
 'Brown blight Tea':"use Copper Oxy Chloride Sprays",
 'Brown spot Rice':"use Si fertilizer",
 'Curl virus cotton':"use lime-sulfur fungicide",
 'Gray spot tomato':"use amino acid sprays",
 'Healthy Rice':"No Need of Fertilizer",
 'Healthy Tea':"No Need of Fertilizer",
 'Healthy cotton':"No Need of Fertilizer",
 'Late blight tomato':"use Mancozeb sprays ",
 'cordana_Banana':"use manganese and copper sprays",
 'fussarium wilt cotton':"use nitrate nitrogen fertilizer",
 'health tomato':"No Need of Fertilizer",
 'healthy_Banana':"No Need of Fertilizer",
 'leaf blast Rice':"Use a protectant fungicide",
 'leaf scald Rice':"use benomyl or fentin acetate sprays",
 'narrow brown spot Rice':"use Remove weeds and weedy rice",
 'pestalotiopsis_Banana':"use foliar sprays of prochloraz",
 'red leaf spot Tea':"use phosphorus or potassium sprays",
 'sigatoka_Banana':"use Timorex Gold",
 'white spot Tea':" use nitrogenous sprays",
    }

    dict2 = {
  'Algal Tea':"typically refers to a nutrient-rich liquid fertilizer made from the infusion of various types of algae. Algae, such as seaweed or freshwater varieties, are known to be rich in essential plant nutrients, growth-promoting hormones, and trace elements.",
 'Anthracnose Tea':"Anthracnose is a group of fungal diseases caused by different species of fungi in the genus Colletotrichum or Glomerella. These fungi are known to affect a wide range of plants, including trees, fruits, vegetables, and ornamental plants. Anthracnose is characterized by dark, sunken lesions on leaves, stems, fruits, or flowers",
 'Bacterial blight cotton':"It is caused by xanthomanas axonopodis. It can be prevented by applying growth regulators.",
 'Bacterial spot tomato':"Bacterial spot of tomato is caused by Xanthomonas vesicatoria.It can be removed by symptomatic plants from the field or greenhouse to prevent the spread of bacteria to healthy plants",
 'Black mold tomato':"It is caused by Alternaria alternata. One way of preventing this disease is to avoid the repeated wetting and drying of the surface of the soil by using subsurface irrigation.",
 'Brown blight Tea':"Brown blight in tea plants is caused by a fungal pathogen called Exobasidium vexans. This pathogen is responsible for a disease commonly known as brown blight or tea blister blight. ",
 'Brown spot Rice':"Brown spot disease in rice is caused by the fungus Cochliobolus miyabeanus. Practicing good sanitation prevents this disease.",
 'Curl virus cotton':"The very characteristic symptoms include leaf curling, darkened veins, vein swelling and enations that frequently develop into cup-shaped, leaf-like structures on the undersides of leaves.",
 'Gray spot tomato':"It is caused by three different fungi,  Stemphylium solani, Stemphylium floridanum, and Stemphylium botryosum. It often strikes in warm, wet weather. Organic and chemical fungicides can be used to prevent this disease.",
 'Healthy Rice':"water it as per required",
 'Healthy Tea':"water it as per required",
 'Healthy cotton':"water it as per required",
 'Late blight tomato':"It is caused by the water Mold Phytophthora infectants. Obtain tomato plants from trusted seed suppliers or reputable local sources to prevent late blight. Copper will kill all of these organisms.",
 'cordana_Banana':"It is called Cordana leaf spot, it is one of the most important fungal diseases of banana which helps in growth.It is caused by fungus.",
 'fussarium wilt cotton':"Leaves on infected plants turn yellow and fall. The plant wilts over several days and then dies. A characteristic symptom of fusarium wilt is the reddish-brown discolouration of the water conducting tissue of the stem and roots",
 'health tomato':"water it as per required",
 'healthy_Banana':"water it as per required",
 'leaf blast Rice':"Blast symptoms appear on leaves as elliptical spots with light-colored centers and reddish edges.The most serious damage from rice blast occurs when the disease attacks the nodes just below the head, often causing the stem to break.",
 'leaf scald Rice':"zonate lesions of alternating light tan and dark brown starting from leaf tips or edges. oblong lesions with light brown halos in mature leaves. translucent leaf tips and margins.",
 'narrow brown spot Rice':"caused by the fungus Cercospora janseana, varies in severity from year to year and is more severe as rice plants approach maturity. Leaf spotting may become very severe on the more susceptible varieties, and the disease causes severe leaf necrosis.",
 'pestalotiopsis_Banana':"The fungus causes leaf spots, petiole/rachis blights and sometimes bud rot of palms. Unlike other leaf spot and blight diseases, Pestalotiopsis palmarun attacks all parts of the leaf from the base to the tip",
 'red leaf spot Tea':"Red leaf spot in tea plants is typically caused by a fungal pathogen known as Cercospora. This disease is commonly referred to as red leaf spot, and it affects the leaves of tea plants. ",
 'sigatoka_Banana':"Black sigatoka (Mycosphaerella fijiensis) first causes small, light yellow spots or streaks on leaves of about one month old. The symptoms run parallel to the veins. Within a few days, the spots become a few centimetres in size and turn brown with light grey centres.",
 'white spot Tea':"Tea plants can be susceptible to various diseases caused by fungi, bacteria, viruses, or environmental factors. "       

    }
   
    
    pesticides = dict[predicted_class]
    discription = dict2[predicted_class]
    disc = discription
    speech = pesticides

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.setProperty('rate', 120)
    audio = engine.say(speech)
    engine.runAndWait()
    

   
    
    return {
        'class': predicted_class,
        'confidence': float(confidence),
        'pesticides':pesticides,
        'disc':disc,
        'speech':audio
   
        
       
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)


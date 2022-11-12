//Dekh bhai me yaha bna rha hu face-api ko use krke ek API zo face recognize kregi and uske corresponding data generate krke use database me store kr rha hu
//yaha pe jitne bhi packages mane install kie ha unko import krdia ha project me
const express = require("express");
const faceapi = require("face-api.js");
const mongoose = require("mongoose");
const { Canvas, Image } = require("canvas");
const canvas = require("canvas");
const fileUpload = require("express-fileupload"); //yaha pe me add kra hu express-fileupload ko as a middleware to our server.
faceapi.env.monkeyPatch({ Canvas, Image });
const app = express();

mongoose.connect("mongodb://localhost:27017/faceDB"); //yaha apne database string use krna ha

app.use(
  fileUpload({
    useTempFiles: true,
  })
);


async function LoadModels() {
  // here i am Loading the pre-trained models of face-api.js package
  // __dirname gives the root directory of the server
  await faceapi.nets.faceRecognitionNet.loadFromDisk(__dirname + "/models");
  await faceapi.nets.faceLandmark68Net.loadFromDisk(__dirname + "/models");
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(__dirname + "/models");
}
LoadModels();

//now we know that we need to store our data in MongoDB database so we need to define a Schema for the collections in  our database
const faceSchema = new mongoose.Schema({
  label: {
    type: String,
    required: true,
    unique: true,
  },
  descriptions: {
    type: Array, // the descriptions as Array will actually contains objects.
    required: true,
  }
});

const FaceModel = mongoose.model("FaceModel", faceSchema);

//NOw we can start receiving labelled face images and store them in the MongoDB database.To do so
//Now we are defining a function that receives a set of images and a label, then extracts the descriptions of the face and stores it in the database
async function uploadLabeledImages(images, label) { //here we are taking the images and label as input into this function
  try { //wraping our function with try-catch so that if there are any errors in the process the app doesn’t crash
    // let counter = 0; //it was just to check my code Progress
    const descriptions = []; // we define an array to store all the descriptions before uploading to the database
    // Looping through the images
    for (let i = 0; i < images.length; i++) { //now we are going through each of the image if present
      const img = await canvas.loadImage(images[i]); // here we are reading the image with canvas.loadImage() function
      // counter = (i / images.length) * 100;
      // console.log(`Progress = ${counter}%`); //just checking the Progress
      // Reading each face and save the face descriptions in the descriptions array using face-api.js method
      const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor(); //Now we have passed in the image data to the face-api methods and detect the faces features
      descriptions.push(detections.descriptor); //Now here The description is extracted from the features and pushed into the descriptions array
    }

    //Now after all the image features are extracted we save the data in the database as per the Schema and return true if the task is completed.
    // Creating a new face document with the given label and save it into our DB
    const createFace = new FaceModel({
      label: label,
      descriptions: descriptions,
    });
    await createFace.save();
    return true;
  } catch (error) {
    console.log(error);
    return (error);
  }
}



async function getDescriptorsFromDB(image) {
  let faces = await FaceModel.find(); //Here we get all the face data from the database
  for (i = 0; i < faces.length; i++) { //Now we have Got all the face data from mongodb and now we are looping through each of them to read the data
    //But the data we get are just objects in arrays Therefore In order for our model to read the descriptions for the image, it needs to be LabeledFaceDescriptors Objects
    // we will Change the face data descriptors from Objects to Float32Array type
    for (j = 0; j < faces[i].descriptions.length; j++) {
      //Therefore,here for each of the face data, we loop through each of the descriptions which are objects. We turn these Object types into Array and then into Float32Array
      faces[i].descriptions[j] = new Float32Array(Object.values(faces[i].descriptions[j])); //However, to do that we need to pass in the descriptions as Float32Array
    }
    faces[i] = new faceapi.LabeledFaceDescriptors(faces[i].label, faces[i].descriptions);
  }

  // Now we can initiate to the facematcher and read the image that has been passed into the function to carry out recognition.
  // Load face matcher to find the matching face
  const faceMatcher = new faceapi.FaceMatcher(faces, 0.6);

  // Read the image using canvas
  // I run the processing and detection functions based on the documentation of the face API and return the results.
  const img = await canvas.loadImage(image);
  let temp = faceapi.createCanvasFromMedia(img);
  // Process the image for the model
  const displaySize = { width: img.width, height: img.height };
  faceapi.matchDimensions(temp, displaySize);

  // Find matching faces
  const detections = await faceapi.detectAllFaces(img).withFaceLandmarks().withFaceDescriptors();
  const resizedDetections = faceapi.resizeResults(detections, displaySize);
  const results = resizedDetections.map((d) => faceMatcher.findBestMatch(d.descriptor));
  return results;
}



//The route is pretty simple. We just receive the files and label, then pass it into the function we defined earlier. Then we send the user a response depending on if it got saved or not.
app.post("/post-face",async (req,res)=>{
    const File1 = req.files.File1.tempFilePath;
    const label = req.body.label;
    let result = await uploadLabeledImages([File1], label);
    if(result){
        res.json({message:"Face data stored successfully"})
    }else{
        res.json({message:"Something went wrong, please try again."})

    }
})


//In this route we just pass in the image and wait for the getDescriptorsFromDB function to carry out face recognition and return the result.
app.post("/check-face", async (req, res) => {

  const File1 = req.files.File1.tempFilePath;
  let result = await getDescriptorsFromDB(File1);
  res.json({ result });

});


app.listen(3000,function(){
  console.log("server is up on port 3000");
});

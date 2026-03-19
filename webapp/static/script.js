function openAbout(){
    document.getElementById("aboutModal").style.display="block"
}

function closeAbout(){
    document.getElementById("aboutModal").style.display="none"
}

async function detectDefect(){

    const fileInput = document.getElementById("imageInput")

    if(fileInput.files.length === 0){
        alert("Please upload an image")
        return
    }

    const file = fileInput.files[0]

    // preview
    const preview = document.getElementById("inputPreview")
    preview.src = URL.createObjectURL(file)
    preview.style.display = "block"

    const formData = new FormData()
    formData.append("file", file)

    try {
        const response = await fetch("/detect",{
            method:"POST",
            body:formData
        })

        if(!response.ok){
            throw new Error("Server error")
        }

        const data = await response.json()

        const resultText = document.getElementById("result")
        resultText.innerText = "Result: " + data.result

        if(data.result.includes("Good")){
            resultText.style.color = "green"
        } else {
            resultText.style.color = "red"
        }

        const resultImage = document.getElementById("resultImage")

        if(data.image){
            resultImage.src = data.image + "?t=" + new Date().getTime()
            resultImage.style.display = "block"
        }

    } catch (error){
        console.error(error)
        alert("Error occurred. Check console.")
    }
}
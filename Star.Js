// Display a greeting message in the console
console.log("Hello, World!");

// Function to add two numbers
function addNumbers(a, b) {
    return a + b;
}

// Call the function and log the result
const result = addNumbers(5, 10);
console.log("The sum is:", result);

// Manipulate the DOM
document.addEventListener("DOMContentLoaded", () => {
    // Select the element with id "message"
    const messageElement = document.getElementById("message");

    // Change the text content of the element
    if (messageElement) {
        messageElement.textContent = "JavaScript is awesome!";
    }
});

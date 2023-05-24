 // Get the form element
 var loginForm = document.getElementById("loginForm");

 // Add an event listener to the form's submit event
 loginForm.addEventListener("submit", function (event) {
   event.preventDefault(); // Prevent the form from submitting
   // Get the input values
   var usernameInput = document.getElementById("inputEmail4");
   var passwordInput = document.getElementById("inputPassword4");

   // Get the entered values
   var username = usernameInput.value;
   var password = passwordInput.value;

   // Perform validation
   if (username.trim() === "") {
	 alert("Please enter your username.");
	 return;
   }

   if (password.trim() === "") {
	 alert("Please enter your password.");
	 return;
   }

   if (password.length < 6) {
    alert("Password must be at least 6 characters long.");
    return;
  }

  if (!/\d/.test(password)) {
    alert("Password must contain at least one digit (0-9).");
    return;
  }

  if (!/[a-z]/i.test(password)) {
    alert("Password must contain at least one letter.");
    return;
  }

   // If validation passes, you can proceed with login logic
   // For now, we'll just display an alert
   alert("Login successful!");
   // You can replace the alert with your own logic for handling the login.

   // Reset the form
   loginForm.reset();
  })
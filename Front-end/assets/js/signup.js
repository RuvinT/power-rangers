document.addEventListener('DOMContentLoaded', () => {
  // Get the signup form elements
  const signUpForm = document.querySelector('.sign-up-htm'); // Get the signup form using its class
  const signUpUsername = document.getElementById('signup-user'); // Get the username input element
  const signUpPassword = document.getElementById('signup-pass'); // Get the password input element
  const signUpRepeatPassword = document.getElementById('signup-repeat-pass'); // Get the repeat password input element
  const signUpEmail = document.getElementById('signup-email'); // Get the email input element
  const signUpSubmit = document.getElementById('signup-submit'); // Get the submit button element

  signUpSubmit.addEventListener('click', (e) => {
    e.preventDefault(); // Prevent the default form submission

    // Get the values from the input elements
    const username = signUpUsername.value;
    const password = signUpPassword.value;
    const repeatPassword = signUpRepeatPassword.value;
    const email = signUpEmail.value;

    var myHeaders = new Headers();
    myHeaders.append("Content-Type", "application/json");

    var raw = JSON.stringify({
      "FullName": "Rajantha",
      "username": username,
      "email": email,
      "password": password,
      "country": "SL",
      "account_cat": "Normal",
      "tc": "1"
    });

    var requestOptions = {
      method: 'POST',
      headers: myHeaders,
      body: raw,
      redirect: 'follow'
    };

    // Validate the form inputs
    if (username === '') {
      alert('Please enter a username.');
      return;
    }

    if (password === '') {
      alert('Please enter a password.');
      return;
    }

    if (password !== repeatPassword) {
      alert('Passwords do not match. Please try again.');
      return;
    }

    if (email === '') {
      alert('Please enter an email address.');
      return;
    }

    fetch("https://uif12sbvza.execute-api.us-east-2.amazonaws.com/EnergySignUp", requestOptions)
    .then(response => response.text())
    .then(result => console.log(result))
    .catch(error => console.log('error', error));

    // If all inputs are valid, display a success message
    alert('Signup successful!');
  });
});

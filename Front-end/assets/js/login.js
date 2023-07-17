document.addEventListener('DOMContentLoaded', function() {
  
  const signInForm = document.querySelector('.sign-in-htm');
  const signInUsername = document.getElementById('signin-user');
  const signInPassword = document.getElementById('signin-pass');
  const signInSubmit = document.getElementById('signin-submit');

  
  signInSubmit.addEventListener('click', function(e) {
    e.preventDefault();

    
    const username = signInUsername.value;
    const password = signInPassword.value;

    if (username === 'valid_username' && password === 'valid_password') {
      alert('Login successful!');
      // Perform any necessary actions after successful login
    } else {
      alert('Invalid username or password. Please try again.');
    }
  });
});

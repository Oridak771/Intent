// Select elements
const play = document.querySelector('.play');
const pause = document.querySelector('.pause');
const playBtn = document.querySelector('.circle__btn');
const wave1 = document.querySelector('.circle__back-1');
const wave2 = document.querySelector('.circle__back-2');

// Initialize states
pause.classList.add('visibility');
play.classList.remove('visibility');
wave1.classList.add('paused');
wave2.classList.add('paused');

// Add event listener to button
playBtn.addEventListener('click', function(e) {
  e.preventDefault();
  pause.classList.toggle('visibility');
  play.classList.toggle('visibility');
  playBtn.classList.toggle('shadow');
  wave1.classList.toggle('paused');
  wave2.classList.toggle('paused');
});

// Toggle functionality (assuming there's a backend API to handle recording state)
const toggleButton = document.querySelector('.components .circle__btn');
toggleButton.addEventListener('click', () => {
  fetch('http://localhost:5000/toggle', { method: 'POST' })
    .then(response => response.json())
    .then(data => {
      console.log(data);
      const isRecording = data.isRecording;
      pause.style.display = isRecording ? 'block' : 'none';
      play.style.display = isRecording ? 'none' : 'block';
    })
    .catch(error => console.error('Error:', error));
});

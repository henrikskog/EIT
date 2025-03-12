function getCounts() {
    setInterval(() => {
      fetch('/counter')
        .then(response => response.json())
        .then(data => {
          console.log("Data from /counter:", data);
          document.getElementById('upCount').textContent = data.upCount;
          document.getElementById('downCount').textContent = data.downCount;
        })
        .catch(err => console.error(err));
    }, 1000);
  }
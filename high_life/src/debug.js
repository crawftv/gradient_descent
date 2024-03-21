const socket = new WebSocket("ws://localhost:8001/ws")
var storedData = localStorage.getItem('tableRows');
var flags = storedData ? JSON.parse(storedData) : null;

const app = Elm.Main.init({
  node: document.getElementById('myapp'),
  flags: flags
});

app.ports.saveTableRows.subscribe(function(tableRows) {
localStorage.setItem('tableRows', JSON.stringify(tableRows));
});
socket.addEventListener("message", function(event) {
    console.log(event.data)
//    app.ports.llmProgressUpdateReceiver.send(event.data);
});
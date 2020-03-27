import React, {useState, useContext} from 'react';
import './App.css';

function App(props) {
  const [state, setState] = useState();


  const styles = {
    backgroundColor: 'yellow',
    backgroundImage: `url("${'logo192.png'}")`,
    width: '192px',
    height: num,
    margin: '10px',
    border: "2px solid blue"
  };
  // js/sandbox/node_modules/react-scripts/scripts/start.js

  const handleChange =
    (e) => setNum(Math.max(Math.min(e.target.valueAsNumber, 192), 192/2));

  return (
    <div className="App" style={styles}>
      <input
        type="number"
        name="num-input"
        onChange={handleChange}
        value={num}
      />
    </div>
  );
}

export default App;

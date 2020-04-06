import React, {useState, useContext} from 'react';
import './App.css';

function App(props) {

  function rainbow(numOfSteps, step) {
    // This function generates vibrant, "evenly spaced" colours (i.e. no clustering). This is ideal for creating easily distinguishable vibrant markers in Google Maps and other apps.
    // Adam Cole, 2011-Sept-14
    // HSV to RBG adapted from: http://mjijackson.com/2008/02/rgb-to-hsl-and-rgb-to-hsv-color-model-conversion-algorithms-in-javascript
    var r, g, b;
    var h = step / numOfSteps;
    var i = ~~(h * 6);
    var f = h * 6 - i;
    var q = 1 - f;
    switch(i % 6){
        case 0: r = 1; g = f; b = 0; break;
        case 1: r = q; g = 1; b = 0; break;
        case 2: r = 0; g = 1; b = f; break;
        case 3: r = 0; g = q; b = 1; break;
        case 4: r = f; g = 0; b = 1; break;
        case 5: r = 1; g = 0; b = q; break;
    }
    var c = "#" + ("00" + (~ ~(r * 255)).toString(16)).slice(-2) + ("00" + (~ ~(g * 255)).toString(16)).slice(-2) + ("00" + (~ ~(b * 255)).toString(16)).slice(-2);
    return (c);
  }
  return

  // replace word elements with perfectly colored boxes which match
  //

  // const [state, setState] = useState();
  //
  //
  // const styles = {
  //   backgroundColor: 'yellow',
  //   backgroundImage: `url("${'logo192.png'}")`,
  //   width: '192px',
  //   height: num,
  //   margin: '10px',
  //   border: "2px solid blue"
  // };
  // // js/sandbox/node_modules/react-scripts/scripts/start.js
  //
  // const handleChange =
  //   (e) => setNum(Math.max(Math.min(e.target.valueAsNumber, 192), 192/2));
  //
  // return (
  //   <div className="App" style={styles}>
  //     <input
  //       type="number"
  //       name="num-input"
  //       onChange={handleChange}
  //       value={num}
  //     />
  //   </div>
  // );
}

export default App;

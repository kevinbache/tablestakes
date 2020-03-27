import React, {useState} from 'react';
// import logo from './logo.svg';
import './App.css';

function App(props) {
  // const defaultStyles = {
  //   backgroundColor: 'blue',
  //   width: '200px',
  //   height: '200px',
  //   margin: '10px',
  // };
  const [num, setNum] = useState(192/2);

  // const randInt = Math.floor(Math.random() * 3);
  // setTimeout(() => setStyles({
  //   backgroundColor: ['blue', 'red', 'green'][randInt],
  //   width: '400px',
  //   height: '400px',
  //   margin: '10px',
  // }), 1000);

  // const handleChange = (e) => {
  //   console.log(e.target.valueAsNumber);
  //   const newNum = Math.max(Math.min(e.target.valueAsNumber, 100), 30);
  //   setNum(newNum);
  // };

  // const [styles, setStyles] = useState();
  // setStyles({
  //   backgroundColor: 'blue',
  //   width: '200px',
  //   height: num,
  //   margin: '10px',
  // });
  const styles = {
    // backgroundColor: 'red',
    backgroundImage: `url("${'/logo192.png'}")`,
    width: '192px',
    height: num,
    margin: '10px',
    border: "2px solid blue"
  };

  return <div className="App" style={styles}>
    <input
      type="number"
      name="num-input"
      onChange={(e) => setNum(Math.max(Math.min(e.target.valueAsNumber, 192), 192/2))}
      value={num}
    />
    {/*<input*/}
    {/*  type='file'*/}
    {/*  onChange={(e) => {*/}
    {/*    console.log('event:');*/}
    {/*    console.log(e);*/}
    {/*    console.log('e.target.value:');*/}
    {/*    console.log(e.target.value);*/}
    {/*    console.log('done');*/}
    {/*    setStyles({*/}
    {/*      ...{backgroundImage: `url(${e.target.files[0].name})`},*/}
    {/*      ...defaultStyles*/}
    {/*    });*/}
    {/*    console.log('e.target.files:');*/}
    {/*    console.log(e.target.files[0]);*/}
    {/*  }}*/}
    {/*/>*/}
  </div>

  // return (
  //   <div className="App">
  //     <header className="App-header">
  //       <img src={logo} className="App-logo" alt="logo" />
  //       <p>
  //         Edit <code>src/App.js</code> and save to reload.
  //       </p>
  //       <a
  //         className="App-link"
  //         href="https://reactjs.org"
  //         target="_blank"
  //         rel="noopener noreferrer"
  //       >
  //         Learn React
  //       </a>
  //     </header>
  //   </div>
  // );
}

export default App;

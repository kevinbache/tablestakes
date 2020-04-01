import React, {useContext, useState} from 'react';
import Papa from "papaparse";
import PropTypes from 'prop-types';
import {BoxesContext} from "../context/BoxesContext";

// https://github.com/kitze/react-hanger
import * as d3 from "d3";


const BoxesPicker = props => {
  const [boxesFilename, setBoxesFilename] = useState('');
  const [boxesColor, setBoxesColor] = useState('blue');
  const [boxesData, setBoxesData] = useState([]);

  const changeFilename = (e) => {
    // console.log('starting changeFilename');
    // console.log('e.target.files: ');
    // console.log(e.target.files);
    const filename = e.target.files[0];
    setBoxesFilename(filename);
    // console.log('finished changeFilename');
    let reader = new FileReader();
    function readFile(event) {
      // console.log('readFile event:');
      // console.log(event);

      const content = event.target.result;
      console.log('readFile content:');
      console.log(content);


      function setDataInState(result) {
        var data = result.data;
        // setState({
        //   state_bak: this.state,
        //   data: data,
        // });
        console.log('papa complete data:');
        console.log(data);
      }

      const results = Papa.parse(content, {
        header: true,
        dynamicTyping: true,
        complete: setDataInState,
      });

      console.log('papa parse final results:');
      console.log(results);


      // const data = event.target.result;
      // console.log('d3 rows:');
      // d3.csv(data).then(function(row) {
      //   console.log(row);
      //   boxesData.push(row);
      // });
    }
    reader.addEventListener('load', readFile);
    reader.readAsText(filename);
    // console.log('text:');
    // console.log(text);

    // // var reader = new FileReader();
    // reader.readAsText(file);

    //
    // function changeFile() {
    //   var file = input.files[0];
    // }


    // console.log('changed boxes filename.  about to read file:');
    // d3.csv(boxesFilename).then(function(row) {
    //   console.log(row);
    // });
    // console.log('done');
  };

  return (
    <div className={"boxesPicker"}>
      <strong>Boxes Picker</strong>

      <form>
        <label htmlFor="boxes-filename">Boxes File:</label>
        <input
          type="file"
          name="boxes-filename"
          id="boxes-filename"
          // value={boxesFilename}
          onChange={ (e) => changeFilename(e) }
        />
        <label htmlFor="color">Boxes Color:</label>
        <input
          type='text'
          name='color'
          id='color'
          onChange={ (e) => setBoxesColor(e.target.value) }
          value={boxesColor}
        />
      </form>
    </div>
  );
};


// const BoxesPicker = props => {
//   const [boxesFilenameInput, boxesData, colorInput, xOffsetInput, xScaleInput, yOffsetInput, yScaleInput] =
//     useContext(BoxesContext);
//
//   const changeFilename = (e) => {
//     try {
//       boxesData.clear();
//       console.log('changed boxes filename.  about to read file:');
//       d3.csv(boxesFilenameInput.value).then(function(row) {
//         boxesData.push(row);
//         console.log(row);
//       });
//       console.log('done');
//     } catch (error) {
//       console.log('error while reading csv file:');
//       console.log(error);
//     }
//
//     return boxesFilenameInput.onChange(e)
//   };
//
//   return (
//     <div className={"boxesPicker"}>
//       Boxes Picker
//
//       <form>
//         <label htmlFor="boxes-filename">Boxes File</label>
//         <input
//           type="file"
//           name="boxes-filename"
//           id="boxes-filename"
//           onChange={ (e) => changeFilename(e) }
//           value={boxesFilenameInput.value}
//         />
//         <label htmlFor="color">Boxes File</label>
//         <input
//           type='text'
//           name='color'
//           id='color'
//           onChange={ (e) => colorInput.onChange(e) }
//           value={colorInput.value}
//         />
//       </form>
//     </div>
//   );
// };



BoxesPicker.propTypes = {

};

export default BoxesPicker;
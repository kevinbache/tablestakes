import React, {useContext, useState} from 'react';
import Papa from "papaparse";
import {BoxesContext} from "../context/BoxesContext";
import {readFileAsText} from "../utils";


const BoxesPicker = props => {
  // const [boxesFilename, setBoxesFilename] = useState('');
  // const [boxesColor, setBoxesColor] = useState('blue');
  // const [boxesData, setBoxesData] = useState([]);
  const [
    boxesFilename, setBoxesFilename,
    boxesColor, setBoxesColor,
    boxesData, setBoxesData,
    xOffsetInput, xScaleInput, yOffsetInput, yScaleInput
  ] = useContext(BoxesContext);

  const changeFilename = (e) => {
    const filename = e.target.files[0];
    setBoxesFilename(filename);

    const parseCsv = (contents) => {
      const results = Papa.parse(contents, {
        header: true,
        dynamicTyping: true,
      });
      setBoxesData(results.data);
    };

    readFileAsText(filename, parseCsv)
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
          onChange={ (e) => changeFilename(e) }
        />

        <label htmlFor="color">Color:</label>
        <input
          type='text'
          name='color'
          id='color'
          onChange={ (e) => setBoxesColor(e.target.value) }
          value={boxesColor}
        />

        <label htmlFor="xOffset">xOffset:</label>
        <input
          type='number'
          name='xOffset'
          id='xOffset'
          onChange={ (e) => xOffsetInput.onChange(e) }
          value={xOffsetInput.value}
        />

        <label htmlFor="xScale">xScale:</label>
        <input
          type='number'
          name='xScale'
          id='xScale'
          onChange={ (e) => xScaleInput.onChange(e) }
          value={xScaleInput.value}
        />

        <label htmlFor="yOffset">yOffset:</label>
        <input
          type='number'
          name='yOffset'
          id='yOffset'
          onChange={ (e) => yOffsetInput.onChange(e) }
          value={yOffsetInput.value}
        />

        <label htmlFor="yScale">yScale:</label>
        <input
          type='number'
          name='yScale'
          id='yScale'
          onChange={ (e) => yScaleInput.onChange(e) }
          value={yScaleInput.value}
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
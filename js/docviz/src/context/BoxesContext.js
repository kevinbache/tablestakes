import React, {useState, createContext} from "react";
import {useArray, useInput} from "react-hanger";

// export class OffsetScale {
//   constructor(xOffset, xScale, yOffset, yScale) {
//     this.xOffset = xOffset;
//     this.xScale = xScale;
//     this.yOffset = yOffset;
//     this.yScale = yScale;
//   }
// }
// export class BoxesState {
//   constructor(boxesData, filename, color, offsetScale) {
//     this.boxesData = boxesData;
//     this.filename = filename;
//     this.color = color;
//     this.offsetScale = offsetScale
//   }
//
//   mutate(objects ) {
//
//   }
// }

export const BoxesContext = createContext();
/*
classes state or many different properties?

with classes state, you can save it and load it.
  write dot object constructor for any change.
  can do multiple changes at once?

with multiple state variables you just pick whatever you need.
  but you end up having to refactor all over?
  harder to coordinate multiple state changes over time
  harder to coordinate with multiple arrays to line up state changes for rewind
 */
export const BoxesProvider = props => {
  const boxesFilenameInput = useInput('');
  const boxesData = useArray([]);
  const colorInput = useInput('rgba(240, 20, 20, 0.2)');

  const xOffsetInput = useInput(0.0);
  const xScaleInput = useInput(1.0);
  const yOffsetInput = useInput(0.0);
  const yScaleInput = useInput(1.0);


  const outputs = [boxesFilenameInput, boxesData, colorInput, xOffsetInput, xScaleInput, yOffsetInput, yScaleInput];
  return (
    <BoxesContext.Provider value={outputs}>
      {props.children}
    </BoxesContext.Provider>
  );
};

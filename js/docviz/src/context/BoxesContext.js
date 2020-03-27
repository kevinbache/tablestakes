import React, {useState, createContext} from "react";

export class OffsetScale {
  constructor(xOffset, xScale, yOffset, yScale) {
    this.xOffset = xOffset;
    this.xScale = xScale;
    this.yOffset = yOffset;
    this.yScale = yScale;
  }
}
export class BoxesState {
  constructor(boxesData, filename, color, offsetScale) {
    this.boxesData = boxesData;
    this.filename = filename;
    this.color = color;
    this.offsetScale = offsetScale
  }
}

export const BoxesContext = createContext();

export const BoxesProvider = props => {
  const [state, setState] = useState([
  ]);

  return (
    <BoxesContext.Provider value={[state, setState]}>
      {props.children}
    </BoxesContext.Provider>
  );
};

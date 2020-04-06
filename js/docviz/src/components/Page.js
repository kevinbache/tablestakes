import React, {useContext, useState} from 'react';
import {PagesContext} from "../context/PagesContext";
import {BoxesContext} from "../context/BoxesContext";


const Page = props => {
  const [imageFilenameInput, scaleInput, pageData, setPageData] =
    useContext(PagesContext);

  const [
    boxesFilename, setBoxesFilename,
    boxesColor, setBoxesColor,
    boxesData, setBoxesData,
    xOffsetInput, xScaleInput, yOffsetInput, yScaleInput
  ] = useContext(BoxesContext);

  const dpi = 500;
  const image_size = [8.5, 11];

  const [width, height] = image_size.map(x => x * dpi * scaleInput.value) ;

  const widthStr = `${width}px`;
  const heightStr = `${height}px`;

  const bg_div_style = {
    backgroundImage: `url('${pageData}')`,
    backgroundSize: "contain",
    backgroundRepeat: "no-repeat",
    width: widthStr,
    height: heightStr,
    border: "0px solid blue",
    position: 'relative',
    padding: '0px',
  };

  const boxesDivs = boxesData.map((boxData) => {
    if (typeof boxData.width === "undefined" || typeof boxData.height === "undefined") {
      return;
    }

    const xOffset = parseFloat(xOffsetInput.value);
    const yOffset = parseFloat(yOffsetInput.value);

    const style = {
      backgroundColor: "rgba(40, 40, 240, 0.2)",
      color:           "rgba(40, 40, 240, 0.0)",
      width:            (boxData.width * xScaleInput.value) * scaleInput.value,
      height:           (boxData.height * yScaleInput.value) * scaleInput.value,
      left:             (boxData.left * xScaleInput.value + xOffset) * scaleInput.value,
      top:              (boxData.top * yScaleInput.value + yOffset) * scaleInput.value,
      position:         'absolute',
      fontSize:         '8.5px',
      border:           '0px solid blue',
      margin:           '0px',
      padding:          '0px',
    };

    return <div style={style} key={boxData.word_id}>{boxData.text}</div>
  });

  return (
    <div id={"page_id"} className={"page"} style={bg_div_style}>
      Page
      {boxesDivs}
    </div>
  );
};

Page.propTypes = {

};

export default Page;
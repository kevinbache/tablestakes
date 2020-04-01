import React, {useContext, useState} from 'react';
import {PagesContext} from "../context/PagesContext";
import imageFile from "../data/doc_01/doc_wrapped_page_0.png";


const Page = props => {
  const [imageFilenameInput, scaleInput] = useContext(PagesContext);

  // const style = {
  //   backgroundImage: imageFilenameInput,
  //
  // };
    const dpi = 500;
    const image_size = [8.5, 11];

    const [width, height] = image_size.map(x => x * dpi * scaleInput.value) ;

    const widthStr = `${width}px`;
    const heightStr = `${height}px`;

    const bg_div_style = {
      backgroundImage: `url('${imageFilenameInput.value}')`,
      backgroundSize: `${widthStr} ${heightStr}`,
      backgroundRepeat: "no-repeat",
      width: widthStr,
      height: heightStr,
      // maxHeight: "90%",
      border: "0px solid blue",
    };


  return (
    <div className={"page"} style={bg_div_style}>
      Page
    </div>
  );
};

Page.propTypes = {

};

export default Page;
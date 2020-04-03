import React, {useState, useContext} from 'react';
import {PagesContext} from "../context/PagesContext";
import {readFileAsDataURL} from "../utils";


const PagePicker = props => {
  const [imageFilenameInput, scaleInput, pageData, setPageData] =
    useContext(PagesContext);

  const clipValue = (value, min, max) => {
    return Math.max(Math.min(value, max), min);
  };

  const handleScaleChange = (e) => {
    e.target.value = clipValue(e.target.value, 0.01, 5.0);
    scaleInput.onChange(e);
  };

  const changeImageFilename = (e) => {
    const filename = e.target.files[0];
    imageFilenameInput.onChange(e);
    readFileAsDataURL(filename, contents => setPageData(contents));
  };

  return (
    <div className={"pagePicker"}>
      Page Picker

      <form>
        <label htmlFor="page-filename">Page File</label>
        <input
          type='file'
          name='page-filename'
          id='page-filename'
          onChange={ (e) => changeImageFilename(e) }
          value={imageFilenameInput.value}
        />

        <label htmlFor='page-scale'>Scale</label>
        <input
          type='number'
          name='page-scale'
          id='page-scale'
          onChange={ (e) => handleScaleChange(e) }
          value={scaleInput.value}
          step={0.01}
        />
      </form>
    </div>
  );
};

PagePicker.propTypes = {

};

export default PagePicker;
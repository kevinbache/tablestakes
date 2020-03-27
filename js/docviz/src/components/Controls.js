// import React, {useState} from 'react';
// import PropTypes from 'prop-types';
import React from 'react';
import PagePicker from "./PagePicker";
import BoxesPicker from "./BoxesPicker";

const Controls = props => {

  return (
    <div className={"controls"}>
      Controls
      <PagePicker/>
      <BoxesPicker/>
    </div>
  );
};

Controls.propTypes = {

};

export default Controls;
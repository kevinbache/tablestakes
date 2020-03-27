import React, {useState} from 'react';
import PropTypes from 'prop-types';
import Viewport from "./Viewport";
import Controls from "./Controls";

const DocViz = props => {

  return (
    <div className={"docViz"}>
      DocViz <br/>
      <Controls/>
      <Viewport/>
    </div>
  );
};

DocViz.propTypes = {

};

export default DocViz;
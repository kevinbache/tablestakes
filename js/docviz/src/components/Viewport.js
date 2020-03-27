import React, {useState} from 'react';
import PropTypes from 'prop-types';
import Page from "./Page";

const Viewport = props => {

  return (
    <div className={"viewport"}>
      Viewport
      <Page/>
    </div>
  );
};

Viewport.propTypes = {

};

export default Viewport;
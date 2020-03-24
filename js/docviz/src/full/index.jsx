import React from 'react';
import ReactDOM from 'react-dom';
// import Papa from 'papaparse';
// import { Series, DataFrame } from 'pandas-js';
import * as d3 from 'd3';

// const fs = require('fs');

// var csv = require('./jquery.csv.js');

class Page extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      stepNumber: 0,
    };
  }

  render() {
    return (
      <div className="page-holder">
        {/*<img className="page-image" src={this.props.image_file} width="100%"></img>*/}
        <Boxes />
      </div>
    );
  }
}

class Boxes extends React.Component {
  render() {
    // // Your parse code, but not seperated in a function
    // var csvFilePath = File("./docviz/selenium_word_locations.csv");
    // var csvFilePath = new File(
    //   fileBits: '',
    //   fileName: "./docviz/selenium_word_locations.csv",
    //   options: {type: "text/plain"},
    // );
    var csvFilePath = "./data/selenium_word_locations.csv";
    // // var csvFile = fs.readFileSync(csvFilePath);
    // var csvFile = csv.toObjects(csvFilePath);
    // var csvFile = Papa.parse(csvFilePath, {
    //     download: true,
    //     step: function(row) {
    //         console.log("Row:", row.data);
    //     },
    //     complete: function() {
    //         console.log("All done!");
    //     }
    // });

    var csvFile = d3.csv(csvFilePath).then(function(row) {
      console.log(row);
    });

    console.log("Whole file:");
    console.log(csvFile);

    // // var Papa = require("papaparse/papaparse.min.js");
    // let data = Papa.parse(csvFile, {
    //   header: true,
    //   download: true,
    //   skipEmptyLines: true,
    //   // // Here this is also available. So we can call our custom class method
    //   // complete: this.updateData,
    // });

    // for (const row of csvFile) {
    //   console.log(row);
    // }

    return <div className="boxes"></div>
  }
}

// ========================================
//
ReactDOM.render(
  // <Page image_file="./docviz/doc_wrapped_page_0.png"/>,
  <Page />,
  document.getElementById('root')
);

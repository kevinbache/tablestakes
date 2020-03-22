import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';

function Square(props) {
  return (
    <button className="square" onClick={props.onClick}>
      {props.value}
    </button>
  );
}

class Board extends React.Component {
  renderSquare(i) {
    return (
      <Square
        value={this.props.squares[i]}
        onClick={() => this.props.onClick(i)}
      />
    );
  }

  renderRow(row_ind) {
      let squares = [];
      for (const col_ind of Array(3).keys()) {
        squares.push(this.renderSquare(row_ind * 3 + col_ind));
      }
      return <div className="board-row">{squares}</div>;
  }

  render() {
    let rows = [];
    for (const row_ind of Array(3).keys()) {
      rows.push(this.renderRow(row_ind));
    }
    return <div>{rows}</div>;
  }
}

class Game extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      history: [{
        squares: Array(9).fill(null),
      }],
      moves: [0],
      stepNumber: 0,
    };
  }

  getNextPlayer(stepNumber = this.state.stepNumber) {
    return stepNumber % 2 ? 'X' : 'O';
  }

  handleBoardClick(i) {
    const history = this.state.history.slice(0, this.state.stepNumber + 1);
    const squares = history[history.length - 1].squares.slice();
    if (calculateWinner(squares) || squares[i]) {
      return;
    }

    const moves = this.state.moves.slice(0, this.state.stepNumber + 1);
    squares[i] = this.getNextPlayer();

    this.setState({
      history: history.concat([{
        squares: squares,
      }]),
      moves: moves.concat([i]),
      stepNumber: history.length,
    });
  }

  arrayInd2Coords(i) {
    return [Math.floor(i / 3), i % 3];
  }

  jumpTo(step) {
    this.setState({
      stepNumber: step,
    });
  }

  render() {
    const board_history = this.state.history.slice(0, this.state.stepNumber + 1);
    const current_squares = board_history[board_history.length - 1].squares;
    const coords_history = this.state.moves;

    const winner = calculateWinner(current_squares);
    let status;
    if (winner) {
      status = 'Winner: ' + winner;
    } else {
      status = 'Next player: ' + this.getNextPlayer();
    }

    const move_list_items = board_history.map( (board, move_ind) => {
      const coord = coords_history[move_ind];
      const step_coords = this.arrayInd2Coords(coord);
      const player = this.getNextPlayer(move_ind + 1);
      const desc = move_ind ?
        'Go to move #' + move_ind + ', (' + player + ' at ' + step_coords + ')':
        'Go to game start';
      return (
        <li key={move_ind}>
          <button onClick={() => this.jumpTo(move_ind)}>{desc}</button>
        </li>
      );
    });

    return (
      <div className="game">
        <div className="game-board">
          <Board
            squares={current_squares}
            onClick={(i) => this.handleBoardClick(i)}
          />
        </div>
        <div className="game-info">
          <div>{status}</div>
          <ol>{move_list_items}</ol>
        </div>
      </div>
    );
  }
}

function calculateWinner(squares) {
  const lines = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
  ];
  for (let i = 0; i < lines.length; i++) {
    const [a, b, c] = lines[i];
    if (squares[a] && squares[a] === squares[b] && squares[a] === squares[c]) {
      return squares[a];
    }
  }
  return null;
}

// ========================================

ReactDOM.render(
  <Game />,
  document.getElementById('root')
);

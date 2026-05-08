// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract FederatedLearning {

    struct ModelUpdate {
        address hospital;
        string modelHash;
        uint256 accuracy;
        uint256 timestamp;
    }

    ModelUpdate[] public updates;

    // ✅ Event for blockchain proof (helps for marks)
    event UpdateSubmitted(
        address indexed hospital,
        string modelHash,
        uint256 accuracy,
        uint256 timestamp
    );

    // ✅ Submit hospital model update (hash + accuracy)
    function submitUpdate(string memory _modelHash, uint256 _accuracy) public {
        updates.push(ModelUpdate(
            msg.sender,
            _modelHash,
            _accuracy,
            block.timestamp
        ));

        emit UpdateSubmitted(msg.sender, _modelHash, _accuracy, block.timestamp);
    }

    // ✅ Returns total number of updates stored
    function getUpdatesCount() public view returns (uint256) {
        return updates.length;
    }

    // ✅ Returns a specific update by index (easy for demo)
    function getUpdate(uint256 index) public view returns (
        address hospital,
        string memory modelHash,
        uint256 accuracy,
        uint256 timestamp
    ) {
        require(index < updates.length, "Invalid index");

        ModelUpdate memory u = updates[index];
        return (u.hospital, u.modelHash, u.accuracy, u.timestamp);
    }
}

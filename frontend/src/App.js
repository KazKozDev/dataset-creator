import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box, Flex } from '@chakra-ui/react';
import Generator from './components/Generator';
import Datasets from './components/Datasets';
import DatasetDetail from './components/DatasetDetail';
import Templates from './components/Templates';
import Quality from './components/Quality';
import Settings from './components/Settings';
import Sidebar from './components/common/Sidebar';
import Dashboard from './components/Dashboard';
import Tasks from './components/Tasks';


function App() {
  return (
    <Flex minH="100vh">
      <Sidebar />

      <Box
        ml="250px"
        width="calc(100% - 250px)"
        p={5}
        overflowY="auto"
        bg="gray.50"
        minH="100vh"
      >
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/generator" element={<Generator />} />
          <Route path="/templates" element={<Templates />} />
          <Route path="/quality" element={<Quality />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/datasets/:id" element={<DatasetDetail />} />
          <Route path="/tasks" element={<Tasks />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Box>
    </Flex>
  );
}

export default App;
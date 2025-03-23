import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box, Flex } from '@chakra-ui/react';

// Import components
import Sidebar from './components/common/Sidebar';
import Dashboard from './components/Dashboard';
import Generator from './components/Generator';
import QualityControl from './components/QualityControl';
import Datasets from './components/Datasets';
import Settings from './components/Settings';
import Tasks from './components/Tasks';
import DatasetDetail from './components/DatasetDetail';

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
          <Route path="/quality" element={<QualityControl />} />
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